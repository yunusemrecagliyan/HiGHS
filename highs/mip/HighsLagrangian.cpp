/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// Lagrangian decomposition for coupling-row (dual-decomposition shape)
// models (clean-room implementation of the textbook method).
//
// Scope: a small row separator R whose removal splits the model into
// blocks coupled only through R. Coupling rows are dualized (priced):
// blocks solve independently per multiplier vector, giving dual bounds,
// and feasible compositions are injected as MIP-start incumbents through
// the native channel. Nothing is ever fixed; the parent MIP always
// produces its own proof. Any anomaly falls back silently.

#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "Highs.h"

namespace {

// Union-find with path halving (local helper, deterministic).
struct DsU {
  std::vector<HighsInt> p;
  DsU() {}
  explicit DsU(HighsInt n) : p(n) {
    for (HighsInt i = 0; i != n; ++i) p[i] = i;
  }
  HighsInt find(HighsInt a) {
    while (p[a] != a) {
      p[a] = p[p[a]];
      a = p[a];
    }
    return a;
  }
  void unite(HighsInt a, HighsInt b) {
    a = find(a);
    b = find(b);
    if (a != b) p[a] = b;
  }
};

}  // namespace

bool HighsMipSolverData::findLagSeparator(
    const HighsLp& model, HighsLagCandidate& cand) const {
  cand = HighsLagCandidate();
  const HighsInt numCol = model.num_col_;
  const HighsInt numRow = model.num_row_;
  if (numCol < 100) {
    cand.reason = "below toy size";
    return false;
  }
  if (model.a_matrix_.format_ != MatrixFormat::kColwise) {
    cand.reason = "matrix not colwise";
    return false;
  }
  std::vector<char> colFixed(numCol, 0);
  for (HighsInt c = 0; c != numCol; ++c) {
    if (model.col_lower_[c] == model.col_upper_[c]) {
      if (!std::isfinite(model.col_lower_[c])) {
        cand.reason = "degenerate fixed column";
        return false;
      }
      colFixed[c] = 1;
    }
  }
  const HighsInt maxCoupling = std::max<HighsInt>(
      1, mipsolver.options_mip_->mip_lagrangian_max_coupling_rows);
  const HighsInt minBlock = 2;  // ranking threshold only; kept blocks may
                                // be smaller (cheap LP subproblems need no
                                // size caps)

  // Row adjacency over unfixed columns (built once; separator rows are
  // skipped during the scans).
  std::vector<HighsInt> rowStart(numRow + 1, 0);
  {
    std::vector<HighsInt> cnt(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el)
        ++cnt[model.a_matrix_.index_[el]];
    }
    for (HighsInt r = 0; r != numRow; ++r)
      rowStart[r + 1] = rowStart[r] + cnt[r];
  }
  std::vector<HighsInt> rowCols(rowStart[numRow], -1);
  {
    std::vector<HighsInt> fill(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el) {
        HighsInt r = model.a_matrix_.index_[el];
        rowCols[rowStart[r] + fill[r]++] = c;
      }
    }
  }
  std::vector<HighsInt> rowDeg(numRow, 0);
  for (HighsInt r = 0; r != numRow; ++r)
    rowDeg[r] = rowStart[r + 1] - rowStart[r];
  HighsInt totalNnz = rowStart[numRow];
  // Candidate cap scales with model size (heuristic): the scan is
  // O(cap * nnz) per round.
  const HighsInt scanCap = std::max<HighsInt>(
      25, std::min<HighsInt>(500, 2000000 / std::max<HighsInt>(1, totalNnz)));

  std::vector<char> inR(numRow, 0);
  HighsInt numR = 0;
  // Pieces of the graph without R (DSU over columns linked by non-R
  // rows).
  auto computePieces = [&](std::vector<std::vector<HighsInt>>& pieces) {
    DsU dsu(numCol);
    for (HighsInt r = 0; r != numRow; ++r) {
      if (inR[r]) continue;
      HighsInt first = -1;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        HighsInt c = rowCols[e];
        if (first < 0)
          first = c;
        else
          dsu.unite(first, c);
      }
    }
    pieces.clear();
    std::vector<HighsInt> rootToPiece(numCol, -1);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c]) continue;
      HighsInt root = dsu.find(c);
      if (rootToPiece[root] < 0) {
        rootToPiece[root] = (HighsInt)pieces.size();
        pieces.emplace_back();
      }
      pieces[rootToPiece[root]].push_back(c);
    }
  };
  // Nontrivial-piece count after additionally removing candidate row.
  auto splitCount = [&](HighsInt excl) -> HighsInt {
    DsU dsu(numCol);
    for (HighsInt r = 0; r != numRow; ++r) {
      if (r == excl || inR[r]) continue;
      HighsInt first = -1;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        HighsInt c = rowCols[e];
        if (first < 0)
          first = c;
        else
          dsu.unite(first, c);
      }
    }
    std::vector<HighsInt> sizes(numCol, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c]) continue;
      ++sizes[dsu.find(c)];
    }
    HighsInt nontrivial = 0;
    for (HighsInt s : sizes)
      if (s >= minBlock) ++nontrivial;
    return nontrivial;
  };
  // Split quality of additionally removing candidate row: the
  // nontrivial-piece count. (Every remaining row touches exactly one DSU
  // piece by construction — its columns were united through it — so no
  // column-span check is needed; the row assignment below re-verifies
  // defensively.)
  // (verifiedSplit merged into splitCount; acceptance uses splitCount.)
  std::vector<std::vector<HighsInt>> pieces;
  for (;;) {
    computePieces(pieces);
    HighsInt curCount = 0;
    HighsInt largest = 0;
    HighsInt largestIdx = -1;
    for (size_t i = 0; i != pieces.size(); ++i) {
      const HighsInt sz = (HighsInt)pieces[i].size();
      if (sz >= minBlock) ++curCount;
      if (sz > largest) {
        largest = sz;
        largestIdx = (HighsInt)i;
      }
    }
    if (largestIdx < 0 || largest < 2 * minBlock) break;
    if (numR >= maxCoupling) break;  // finalization decides validity
    // Rank candidate rows touching the largest piece.
    std::vector<char> inLargest(numCol, 0);
    for (HighsInt c : pieces[largestIdx]) inLargest[c] = 1;
    HighsInt scanned = 0;
    HighsInt topR[3] = {-1, -1, -1};
    HighsInt topN[3] = {-1, -1, -1};
    for (HighsInt r = 0; r != numRow && scanned < scanCap; ++r) {
      if (inR[r]) continue;
      if (rowDeg[r] < 2 || rowDeg[r] > 64) continue;
      bool touches = false;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        if (inLargest[rowCols[e]]) {
          touches = true;
          break;
        }
      }
      if (!touches) continue;
      ++scanned;
      const HighsInt q = splitCount(r);
      for (HighsInt t = 0; t != 3; ++t) {
        if (q > topN[t]) {
          for (HighsInt u = 2; u != t; --u) {
            topN[u] = topN[u - 1];
            topR[u] = topR[u - 1];
          }
          topN[t] = q;
          topR[t] = r;
          break;
        }
      }
    }
    bool accepted = false;
    for (HighsInt t = 0; t != 3; ++t) {
      if (topR[t] < 0) break;
      const HighsInt q = splitCount(topR[t]);
      if (q >= 2 && q > curCount) {
        inR[topR[t]] = 1;
        ++numR;
        accepted = true;
        if (mipsolver.options_mip_->mip_decomposition_logging)
          highsLogUser(mipsolver.options_mip_->log_options,
                       HighsLogType::kInfo,
                       "[Lag] separator: add row %d (pieces %d -> %d)\n",
                       (int)topR[t], (int)curCount, (int)q);
        break;
      }
    }
    if (!accepted) break;
  }
  // Collect blocks: every piece becomes a block (no size merge; LP
  // subproblems are cheap at any size). Rowless columns merge into one
  // trivial analytic block.
  {
    DsU dsu(numCol);
    for (HighsInt r = 0; r != numRow; ++r) {
      if (inR[r]) continue;
      HighsInt first = -1;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        HighsInt c = rowCols[e];
        if (first < 0)
          first = c;
        else
          dsu.unite(first, c);
      }
    }
    std::vector<HighsInt> rootToPiece(numCol, -1);
    std::vector<HighsInt> rowless;
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c]) continue;
      bool hasRow = false;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el) {
        HighsInt r = model.a_matrix_.index_[el];
        if (!inR[r]) {
          hasRow = true;
          break;
        }
      }
      if (!hasRow) {
        rowless.push_back(c);
        continue;
      }
      HighsInt root = dsu.find(c);
      if (rootToPiece[root] < 0) {
        rootToPiece[root] = (HighsInt)cand.blockCols.size();
        cand.blockCols.emplace_back();
      }
      cand.blockCols[rootToPiece[root]].push_back(c);
    }
    if (!rowless.empty()) cand.blockCols.push_back(std::move(rowless));
  }
  if ((HighsInt)cand.blockCols.size() < 2) {
    cand.reason = "fewer than two blocks";
    return false;
  }
  for (HighsInt r = 0; r != numRow; ++r)
    if (inR[r]) cand.couplingRows.push_back(r);
  // Assign rows: non-separator rows touching exactly one block are block
  // rows; anything else is a separator bug: reject instead of risking an
  // invalid decomposition.
  std::vector<HighsInt> blockOf(numCol, -1);
  for (size_t k = 0; k != cand.blockCols.size(); ++k)
    for (HighsInt c : cand.blockCols[k]) blockOf[c] = (HighsInt)k;
  cand.blockRows.assign(cand.blockCols.size(), {});
  std::vector<HighsInt> seenBlock;
  seenBlock.reserve(8);
  for (HighsInt r = 0; r != numRow; ++r) {
    if (inR[r]) continue;
    seenBlock.clear();
    for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
      HighsInt c = rowCols[e];
      HighsInt b = blockOf[c];
      if (b < 0) continue;  // fixed column (not in rowCols by construction)
      if (std::find(seenBlock.begin(), seenBlock.end(), b) ==
          seenBlock.end())
        seenBlock.push_back(b);
    }
    if (seenBlock.empty()) {
      // Row with no block columns (fully fixed or empty): determined
      // activity, checked by the caller.
      continue;
    } else if (seenBlock.size() == 1) {
      cand.blockRows[seenBlock[0]].push_back(r);
    } else {
      cand = HighsLagCandidate();
      cand.reason = "row spans two blocks (separator bug)";
      return false;
    }
  }
  cand.valid = true;
  cand.reason = "ok";
  return true;
}

bool HighsMipSolverData::runLagrangian() {
  HighsLp& model = presolvedModel;
  const HighsInt numCol = model.num_col_;
  const HighsInt numRow = model.num_row_;
  if (numCol == 0 || numRow == 0) return true;
  if (numCol < 100) return true;
  if (!mipsolver.options_mip_->mip_decomposition) return true;
  if (!mipsolver.options_mip_->mip_lagrangian) return true;
  if (model.a_matrix_.format_ != MatrixFormat::kColwise)
    model.a_matrix_.ensureColwise();
  const bool logLag = mipsolver.options_mip_->mip_decomposition_logging;
  const HighsLogOptions& logOptions = mipsolver.options_mip_->log_options;
  const double feastol = mipsolver.options_mip_->mip_feasibility_tolerance;

  HighsLagCandidate cand;
  if (!findLagSeparator(model, cand)) {
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] no candidate (%s) -> normal MIP\n",
                   cand.reason.c_str());
    return true;
  }
  // Internal minimization (same single-convention pattern as Benders).
  const double sign =
      (model.sense_ == ObjSense::kMaximize) ? -1.0 : 1.0;
  const HighsInt nB = (HighsInt)cand.blockCols.size();

  std::vector<char> colFixed(numCol, 0);
  for (HighsInt c = 0; c != numCol; ++c) {
    if (model.col_lower_[c] == model.col_upper_[c]) {
      if (!std::isfinite(model.col_lower_[c])) return true;
      colFixed[c] = 1;
    }
  }
  std::vector<char> inR(numRow, 0);
  for (HighsInt r : cand.couplingRows) inR[r] = 1;
  // Fixed-column activity shifted out of every row (exact: lb == ub).
  std::vector<double> rowShift(numRow, 0.0);
  for (HighsInt c = 0; c != numCol; ++c) {
    if (!colFixed[c]) continue;
    const double fixval = model.col_lower_[c];
    for (HighsInt el = model.a_matrix_.start_[c];
         el != model.a_matrix_.start_[c + 1]; ++el)
      rowShift[model.a_matrix_.index_[el]] +=
          model.a_matrix_.value_[el] * fixval;
  }
  // Fully-determined rows must hold; a violation proves global
  // infeasibility (nothing left to decide in them). Linear scan: mark
  // rows touched by unfixed columns, check the rest.
  {
    std::vector<char> rowTouched(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el)
        rowTouched[model.a_matrix_.index_[el]] = 1;
    }
    for (HighsInt r = 0; r != numRow; ++r) {
      if (rowTouched[r]) continue;
      if (rowShift[r] < model.row_lower_[r] - feastol ||
          rowShift[r] > model.row_upper_[r] + feastol) {
        mipsolver.modelstatus_ = HighsModelStatus::kInfeasible;
        return true;
      }
    }
  }

  // Directed penalty arcs: <= rows contribute lambda*(act-U),
  // >= rows lambda*(L-act), equalities lambda*(act-b); ranged rows
  // split into both parts. Multipliers of <=/>= parts stay nonneg,
  // equality multipliers are free.
  struct LagArc {
    HighsInt row;
    HighsInt dir;  // +1: <= part, -1: >= part, 0: equality
    double bound;  // shifted U / L / b in the penalty constant
  };
  std::vector<LagArc> arcs;
  for (HighsInt r : cand.couplingRows) {
    const double lo = model.row_lower_[r] == -kHighsInf
                          ? -kHighsInf
                          : model.row_lower_[r] - rowShift[r];
    const double hi = model.row_upper_[r] == kHighsInf
                          ? kHighsInf
                          : model.row_upper_[r] - rowShift[r];
    const bool hasLo = lo != -kHighsInf;
    const bool hasHi = hi != kHighsInf;
    if (!hasLo && !hasHi) continue;  // free row: no penalty possible
    if (hasLo && hasHi && lo == hi) {
      arcs.push_back({r, 0, lo});
    } else {
      if (hasHi) arcs.push_back({r, +1, hi});
      if (hasLo) arcs.push_back({r, -1, lo});
    }
  }
  if (arcs.empty()) {
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] %s -> normal MIP\n",
                   cand.couplingRows.empty()
                       ? "no separator rows (decoupled case belongs to "
                         "Benders/components)"
                       : "coupling rows carry no finite bounds");
    return true;
  }
  const HighsInt nA = (HighsInt)arcs.size();
  std::vector<std::vector<HighsInt>> rowArcs(numRow);
  for (HighsInt a = 0; a != nA; ++a) rowArcs[arcs[a].row].push_back(a);

  struct LagBlock {
    std::vector<HighsInt> cols;
    std::vector<HighsInt> rows;
    // Per block-column: (arc index, matrix coefficient) for cost
    // modification. Coupling rows are never block rows.
    std::vector<std::vector<std::pair<HighsInt, double>>> colArcs;
    std::vector<double> baseLo;
    std::vector<double> baseHi;
    std::vector<double> cost;  // internal-min costs
    std::vector<double> lb;
    std::vector<double> ub;
    bool isLp = true;
  };
  std::vector<LagBlock> blocks(nB);
  bool separatorBroken = false;
  for (HighsInt k = 0; k != nB && !separatorBroken; ++k) {
    LagBlock& blk = blocks[k];
    blk.cols = cand.blockCols[k];
    blk.rows = cand.blockRows[k];
    blk.colArcs.assign(blk.cols.size(), {});
    blk.baseLo.resize(blk.rows.size());
    blk.baseHi.resize(blk.rows.size());
    blk.cost.resize(blk.cols.size());
    blk.lb.resize(blk.cols.size());
    blk.ub.resize(blk.cols.size());
    std::vector<HighsInt> rowToPos(numRow, -1);
    for (size_t i = 0; i != blk.rows.size(); ++i)
      rowToPos[blk.rows[i]] = (HighsInt)i;
    for (size_t j = 0; j != blk.cols.size(); ++j) {
      HighsInt c = blk.cols[j];
      blk.cost[j] = sign * model.col_cost_[c];
      blk.lb[j] = model.col_lower_[c];
      blk.ub[j] = model.col_upper_[c];
      if (model.integrality_[c] != HighsVarType::kContinuous) blk.isLp = false;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el) {
        HighsInt r = model.a_matrix_.index_[el];
        const double v = model.a_matrix_.value_[el];
        if (inR[r]) {
          for (HighsInt a : rowArcs[r]) blk.colArcs[j].emplace_back(a, v);
        } else if (rowToPos[r] < 0) {
          separatorBroken = true;
          break;
        }
      }
    }
  }
  if (separatorBroken) {
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] block column touches foreign row -> normal MIP\n");
    return true;
  }
  for (HighsInt k = 0; k != nB; ++k) {
    LagBlock& blk = blocks[k];
    for (size_t i = 0; i != blk.rows.size(); ++i) {
      HighsInt r = blk.rows[i];
      blk.baseLo[i] = model.row_lower_[r] == -kHighsInf
                          ? -kHighsInf
                          : model.row_lower_[r] - rowShift[r];
      blk.baseHi[i] = model.row_upper_[r] == kHighsInf
                          ? kHighsInf
                          : model.row_upper_[r] - rowShift[r];
    }
  }

  HighsInt numLpBlocks = 0;
  HighsInt blockCols = 0;
  for (HighsInt k = 0; k != nB; ++k) {
    numLpBlocks += blocks[k].isLp ? 1 : 0;
    blockCols += (HighsInt)blocks[k].cols.size();
  }
  const HighsInt maxIter = std::max<HighsInt>(
      1, mipsolver.options_mip_->mip_lagrangian_max_iterations);
  const double maxTime = mipsolver.options_mip_->mip_lagrangian_max_time;
  if (logLag)
    highsLogUser(logOptions, HighsLogType::kInfo,
                 "[Lag] candidate: %d coupling rows (%d arcs), %d blocks "
                 "(%d LP, %d MIP-relaxed, %d block cols)\n",
                 (int)cand.couplingRows.size(), (int)nA, (int)nB,
                 (int)numLpBlocks, (int)(nB - numLpBlocks), (int)blockCols);

  // Subgradient loop. Per-arc multiplier lambda with direction sigma:
  // <= parts and equalities sigma=+1, >= parts sigma=-1, so the modified
  // cost is c + sum lambda*sigma*a and the penalty constant is
  // -lambda*U / +lambda*L / -lambda*b respectively.
  std::vector<double> lambda(nA, 0.0);
  std::vector<double> sigma(nA, 1.0);
  for (HighsInt a = 0; a != nA; ++a)
    sigma[a] = (arcs[a].dir < 0) ? -1.0 : 1.0;
  double bestLB = -kHighsInf;  // internal space
  double bestUB = kHighsInf;
  std::vector<double> bestSol;
  bool hasUB = false;
  HighsInt numIter = 0;
  bool converged = false;
  const double lagStart = mipsolver.timer_.getWallTime();
  std::vector<std::vector<double>> blockSol(nB);
  for (HighsInt iter = 0; iter != maxIter; ++iter) {
    if (mipsolver.options_mip_->time_limit < kHighsInf &&
        mipsolver.timer_.read() >= mipsolver.options_mip_->time_limit)
      break;
    if (maxTime < kHighsInf &&
        mipsolver.timer_.getWallTime() - lagStart >= maxTime)
      break;
    // Modified costs for this multiplier vector.
    double lagLB = 0.0;
    for (HighsInt k = 0; k != nB; ++k) {
      const LagBlock& blk = blocks[k];
      const HighsInt nbC = (HighsInt)blk.cols.size();
      const HighsInt nbR = (HighsInt)blk.rows.size();
      HighsLp sublp;
      sublp.num_col_ = nbC;
      sublp.num_row_ = nbR;
      sublp.sense_ = ObjSense::kMinimize;
      sublp.offset_ = 0.0;
      sublp.a_matrix_.format_ = MatrixFormat::kColwise;
      sublp.a_matrix_.start_.assign(nbC + 1, 0);
      sublp.col_cost_.resize(nbC);
      sublp.col_lower_ = blk.lb;
      sublp.col_upper_ = blk.ub;
      // MIP blocks solve their LP relaxation here: valid (weaker) dual
      // bounds at simplex cost. Integrality only matters for primal
      // compositions, which are verified independently.
      sublp.integrality_.assign(nbC, HighsVarType::kContinuous);
      for (HighsInt j = 0; j != nbC; ++j) {
        double cj = blk.cost[j];
        for (const auto& e : blk.colArcs[j])
          cj += lambda[e.first] * sigma[e.first] * e.second;
        sublp.col_cost_[j] = cj;
      }
      sublp.row_lower_ = blk.baseLo;
      sublp.row_upper_ = blk.baseHi;
      // Block-row entries come straight from the parent matrix.
      std::vector<HighsInt> rowPos(numRow, -1);
      for (HighsInt i = 0; i != nbR; ++i) rowPos[blk.rows[i]] = i;
      for (HighsInt j = 0; j != nbC; ++j) {
        HighsInt c = blk.cols[j];
        for (HighsInt el = model.a_matrix_.start_[c];
             el != model.a_matrix_.start_[c + 1]; ++el) {
          HighsInt sr = rowPos[model.a_matrix_.index_[el]];
          if (sr < 0) continue;  // separator-row entry (dualized)
          sublp.a_matrix_.index_.push_back(sr);
          sublp.a_matrix_.value_.push_back(model.a_matrix_.value_[el]);
        }
        sublp.a_matrix_.start_[j + 1] =
            (HighsInt)sublp.a_matrix_.index_.size();
      }
      if (nbR == 0) {
        // Rowless block: analytic bound minimum over the box.
        std::vector<double> sol(nbC, 0.0);
        double val = 0.0;
        bool bounded = true;
        for (HighsInt j = 0; j != nbC; ++j) {
          const double cj = sublp.col_cost_[j];
          if (cj > 0) {
            if (!std::isfinite(sublp.col_lower_[j])) {
              bounded = false;
              break;
            }
            sol[j] = sublp.col_lower_[j];
            val += cj * sol[j];
          } else if (cj < 0) {
            if (!std::isfinite(sublp.col_upper_[j])) {
              bounded = false;
              break;
            }
            sol[j] = sublp.col_upper_[j];
            val += cj * sol[j];
          } else {
            sol[j] = std::isfinite(sublp.col_lower_[j])
                         ? sublp.col_lower_[j]
                         : 0.0;
          }
        }
        if (!bounded) return true;  // dual unbounded: fallback
        blockSol[k] = std::move(sol);
        lagLB += val;
        continue;
      }
      double remaining =
          mipsolver.options_mip_->time_limit - mipsolver.timer_.read();
      HighsSubLpResult res =
          solveSubLp(sublp, std::min(10.0, remaining));
      if (logLag) {
        // Independent box-minimum check (theorem litmus): no row set can
        // push a minimum below the bound-only minimum.
        double boxmin = 0.0;
        bool boxbounded = true;
        for (HighsInt j = 0; j != nbC; ++j) {
          const double cj = sublp.col_cost_[j];
          if (cj > 0) {
            if (!std::isfinite(sublp.col_lower_[j])) {
              boxbounded = false;
              break;
            }
            boxmin += cj * sublp.col_lower_[j];
          } else if (cj < 0) {
            if (!std::isfinite(sublp.col_upper_[j])) {
              boxbounded = false;
              break;
            }
            boxmin += cj * sublp.col_upper_[j];
          }
        }
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Lag] block %d sublp: status=%d obj=%.6g boxmin=%.6g%s\n",
                     (int)k, (int)res.status, res.obj, boxmin,
                     (boxbounded && res.obj < boxmin - 1e-6 * fabs(boxmin) - 1e-9)
                         ? " VIOLATION" : "");
      }
      if (res.status == HighsModelStatus::kOptimal) {
        if ((HighsInt)res.colSol.size() != nbC) return true;
        blockSol[k] = res.colSol;
        lagLB += res.obj;
      } else if (res.status == HighsModelStatus::kInfeasible) {
        // Block rows alone infeasible: the relaxation is infeasible, so
        // the true block (and hence the whole model) is infeasible.
        mipsolver.modelstatus_ = HighsModelStatus::kInfeasible;
        if (logLag)
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[Lag] block %d infeasible without coupling -> "
                       "globally infeasible\n",
                       (int)k);
        return true;
      } else {
        return true;  // unbounded subproblem or solver trouble: fallback
      }
    }
    // Penalty constants complete the dual bound.
    for (HighsInt a = 0; a != nA; ++a) {
      if (arcs[a].dir > 0)
        lagLB += -lambda[a] * arcs[a].bound;
      else if (arcs[a].dir < 0)
        lagLB += lambda[a] * arcs[a].bound;
      else
        lagLB += -lambda[a] * arcs[a].bound;
    }
    if (lagLB > bestLB) bestLB = lagLB;
    // Subgradient = coupling-row violations at the block solutions.
    std::vector<double> activity(numRow, 0.0);
    for (HighsInt k = 0; k != nB; ++k) {
      for (size_t j = 0; j != blocks[k].cols.size(); ++j) {
        HighsInt c = blocks[k].cols[j];
        const double v = blockSol[k][j];
        for (HighsInt el = model.a_matrix_.start_[c];
             el != model.a_matrix_.start_[c + 1]; ++el)
          activity[model.a_matrix_.index_[el]] +=
              model.a_matrix_.value_[el] * v;
      }
    }
    double gnorm2 = 0.0;
    std::vector<double> grad(nA, 0.0);
    for (HighsInt a = 0; a != nA; ++a) {
      HighsInt r = arcs[a].row;
      double act = activity[r];
      // Shifted frame: fixed activity already removed from bounds, and
      // block solutions cover all unfixed columns of non-separator rows.
      // Separator rows may additionally touch fixed columns only, which
      // rowShift accounts for via the shifted bound stored in the arc.
      double viol = 0.0;
      if (arcs[a].dir > 0)
        viol = act - arcs[a].bound;
      else if (arcs[a].dir < 0)
        viol = arcs[a].bound - act;
      else
        viol = act - arcs[a].bound;
      grad[a] = viol;
      gnorm2 += viol * viol;
    }
    // Primal attempt: the composition may already satisfy the coupling
    // rows (e.g. loose coupling); only then is there anything to inject.
    bool couplingOk = true;
    for (HighsInt a = 0; a != nA; ++a) {
      const double tol = feastol * std::max(1.0, std::fabs(arcs[a].bound));
      if (arcs[a].dir > 0) {
        if (grad[a] > tol) {
          couplingOk = false;
          break;
        }
      } else if (arcs[a].dir < 0) {
        if (grad[a] > tol) {
          couplingOk = false;
          break;
        }
      } else {
        if (std::fabs(grad[a]) > tol) {
          couplingOk = false;
          break;
        }
      }
    }
    if (couplingOk) {
      std::vector<double> fullSol(numCol, 0.0);
      for (HighsInt c = 0; c != numCol; ++c)
        fullSol[c] = model.col_lower_[c];
      for (HighsInt k = 0; k != nB; ++k) {
        for (size_t j = 0; j != blocks[k].cols.size(); ++j)
          fullSol[blocks[k].cols[j]] = blockSol[k][j];
      }
      if (verifyBendersSolution(model, fullSol)) {
        double composed = 0.0;
        for (HighsInt c = 0; c != numCol; ++c)
          composed += sign * model.col_cost_[c] * fullSol[c];
        if (composed < bestUB) {
          bestUB = composed;
          bestSol = std::move(fullSol);
          hasUB = true;
        }
      }
    }
    ++numIter;
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] iter %d: LB=%.6g UB=%.6g gnorm=%.3g\n", (int)iter,
                   bestLB, bestUB, std::sqrt(gnorm2));
    const double gapTol = 1e-7 * std::max(1.0, std::fabs(bestUB));
    if (hasUB && bestUB - bestLB <= gapTol) {
      converged = true;
      break;
    }
    if (gnorm2 < 1e-24) break;  // stationary: gap decides fix vs fallback
    // Multiplier step: Polyak when a UB exists, else normalized
    // diminishing (heuristic step length, documented).
    double step;
    if (hasUB && bestUB - bestLB > 0)
      step = (bestUB - bestLB) / gnorm2;
    else
      step = 1.0 / (std::sqrt(1.0 + (double)iter) * std::sqrt(gnorm2));
    for (HighsInt a = 0; a != nA; ++a) {
      lambda[a] += step * grad[a];
      if (arcs[a].dir != 0 && lambda[a] < 0.0) lambda[a] = 0.0;
    }
  }

  const double parentLB = sign * bestLB + model.offset_;
  if (!hasUB) {
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] no feasible composition (%d iters, best dual "
                   "bound %.6g) -> normal MIP\n",
                   (int)numIter, parentLB);
    return true;
  }
  if (!converged) {
    // Gap open: the incumbent below may still improve the parent search,
    // but only a converged loop proves anything; inject solely as a
    // verified MIP start (never as a fixing).
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] gap open after %d iters; offering incumbent\n",
                   (int)numIter);
  }
  // Inject the best verified composition through the native MIP-start
  // channel: postsolve to original space, re-check against the original
  // model, and publish only if feasible there. checkAddSolution picks it
  // up in runSetup; infeasible candidates are silently dropped.
  //
  // NOTE: pass nullptr (not hand-built row values) as pass_row_value:
  // solutionFeasible trusts a provided row vector verbatim, and only a
  // freshly recomputed activity vector is valid after postsolve.
  HighsSolution injsol;
  injsol.col_value = bestSol;
  injsol.value_valid = true;
  injsol.dual_valid = false;
  HighsBasis injbasis;
  injbasis.valid = false;
  // NOTE: thread_safe=false is the production path (used at every solve
  // end). Calling it mid-presolve is safe: it resets its cursor and only
  // reads the stacks, so the final postsolve re-walks identically. Full
  // ctest (exact solution checks) guards this claim.
  postSolveStack.undo(*mipsolver.options_mip_, injsol, injbasis, -1, false);
  double boundViol = kHighsInf, rowViol = kHighsInf, intViol = kHighsInf;
  HighsCDouble injObj = 0.0;
  mipsolver.solutionFeasible(mipsolver.orig_model_, injsol.col_value,
                             nullptr, boundViol, rowViol, intViol, injObj);
  if (boundViol <= feastol && rowViol <= feastol && intViol <= feastol) {
    mipsolver.solution_ = injsol.col_value;
    mipsolver.solution_objective_ = double(injObj);
    mipsolver.bound_violation_ = boundViol;
    mipsolver.row_violation_ = rowViol;
    mipsolver.integrality_violation_ = intViol;
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] injected incumbent (obj %.6g, dual bound %.6g)\n",
                   double(injObj), parentLB);
  } else if (logLag) {
    highsLogUser(logOptions, HighsLogType::kInfo,
                 "[Lag] postsolved incumbent infeasible (%.2g, %.2g, %.2g) "
                 "-> dropped\n",
                 boundViol, rowViol, intViol);
  }
  return true;
}

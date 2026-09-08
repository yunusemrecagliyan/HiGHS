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
//
// Dual-bound note (measured + proven, do NOT inject this bound into
// B&B): with LP-relaxed blocks the Lagrangian dual D(lam) sits below
// the root LP bound for every multiplier (block-feasible set contains
// the root-feasible set; the root LP point prices out non-positive),
// so presolve-time injection can at best tie the root LP and never
// prune. Only integer-subproblem duals (Benders side) can beat it.

#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Highs.h"

namespace {

// Decomposition master-switch mode ("on"/"off"/"auto"): off skips, on
// forces full budgets (historical behavior), auto probes the first
// iteration cheaply and aborts early on stall. Legacy booleans ("true"
// behaves as on, "false" as off) keep working; anything else fails
// closed to off (caller logs once).
enum class LagDecompMode { Off, On, Auto };
static LagDecompMode parseLagDecompMode(const std::string& value) {
  if (value == "on" || value == "true" || value == "1")
    return LagDecompMode::On;
  if (value == "off" || value == "false" || value == "0")
    return LagDecompMode::Off;
  if (value == "auto") return LagDecompMode::Auto;
  return LagDecompMode::Off;
}

// Union-find with path halving (local helper, deterministic).
struct LagDsU {
  std::vector<HighsInt> p;
  LagDsU() {}
  explicit LagDsU(HighsInt n) : p(n) {
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
  HighsInt scanCap = mipsolver.options_mip_->mip_lagrangian_scan_cap;
  if (scanCap <= 0) {
    scanCap = numRow;
  }

  std::vector<char> inR(numRow, 0);
  HighsInt numR = 0;
  // Pieces of the graph without R (DSU over columns linked by non-R
  // rows).
  auto computePieces = [&](std::vector<std::vector<HighsInt>>& pieces) {
    LagDsU dsu(numCol);
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
    LagDsU dsu(numCol);
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
    const HighsInt maxRowDeg = mipsolver.options_mip_->mip_lagrangian_max_row_degree;
    for (HighsInt r = 0; r != numRow && scanned < scanCap; ++r) {
      if (inR[r]) continue;
      if (rowDeg[r] < 2 || rowDeg[r] > maxRowDeg) continue;
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
    LagDsU dsu(numCol);
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
  const std::string& lagOpt = mipsolver.options_mip_->mip_lagrangian;
  const LagDecompMode lagMode = parseLagDecompMode(lagOpt);
  if (lagMode == LagDecompMode::Off) {
    if (lagOpt != "off")
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kWarning,
                   "Unknown mip_lagrangian value '%s' (want on/off/auto): "
                   "Lagrangian disabled\n",
                   lagOpt.c_str());
    return true;
  }
  const bool lagAuto = (lagMode == LagDecompMode::Auto);
  if (lagAuto && lagProbeFailed) return true;
  const double lagProbe = std::max(
      0.0, mipsolver.options_mip_->mip_lagrangian_probe_time);
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
  // Single-evaluation probe (dual-shape mapping): start all multipliers
  // at the fixed value instead of zero.
  const double fixedLambda =
      mipsolver.options_mip_->mip_lagrangian_fixed_lambda;
  if (fixedLambda >= 0.0) {
    std::fill(lambda.begin(), lambda.end(), fixedLambda);
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] fixed-lambda probe at %.6g (all %d arcs)\n",
                   fixedLambda, (int)nA);
  }
  double bestLB = -kHighsInf;  // internal space
  double bestUB = kHighsInf;
  std::vector<double> bestSol;
  bool hasUB = false;
  HighsInt numIter = 0;
  bool converged = false;
  const double lagStart = mipsolver.timer_.getWallTime();
  // Auto price sweep (primal Lagrangian): fixed_lambda < 0 selects it.
  // The first iterations evaluate a small price set instead of ascending
  // from zero; the winner (best feasible composition, else best bound)
  // seeds the ascent below. Motivation: prices coordinate blocks into
  // coupling-feasible compositions (Salihli 5/5 Optimal at root from the
  // 0.5 leg), while ascent from zero sits flat on integer blocks.
  // Initial pass only (like the repair solves): post-restart models are
  // LP relaxations plus cuts, where the separator may be a cut artifact,
  // and the sweep costs an evaluation per leg.
  const bool autoSweep = fixedLambda < 0.0 &&
                         mipsolver.options_mip_->mip_lagrangian_auto_lambda &&
                         numRestarts == 0;
  const double sweepLams[3] = {0.0, 0.5, 2.0};
  std::vector<double> sweepUBLam = lambda;
  std::vector<double> sweepLBLam = lambda;
  double sweepBestUB = kHighsInf;
  double sweepBestLB = -kHighsInf;
  bool sweepHasUB = false;
  bool sweepDone = !autoSweep;
  std::vector<std::vector<double>> blockSol(nB);
  for (HighsInt iter = 0; iter != maxIter; ++iter) {
    // Auto-probe verdict counters, reset every iteration (only iter 0
    // is judged).
    HighsInt numAttempted = 0;  // solver-backed blocks this iteration
    HighsInt numOptimal = 0;    // ... proven optimal
    if (mipsolver.options_mip_->time_limit < kHighsInf &&
        mipsolver.timer_.read() >= mipsolver.options_mip_->time_limit)
      break;
    // Auto mode probes the first iteration cheaply (later iterations
    // and "on" mode use full budgets; a zero probe disables probing).
    // Sub-MIP blocks keep their tight historical 2s cap regardless:
    // probing targets degenerate LP stalls, not MIP search.
    const double iterLpCap = (lagAuto && iter == 0 && lagProbe > 0.0)
                                 ? lagProbe
                                 : 10.0;
    if (maxTime < kHighsInf &&
        mipsolver.timer_.getWallTime() - lagStart >= maxTime)
      break;
    // Sweep legs override the multipliers; afterwards the winner seeds
    // the ascent (restored once, on the first non-sweep iteration).
    if (!sweepDone) {
      if (iter < 3) {
        std::fill(lambda.begin(), lambda.end(), sweepLams[(int)iter]);
        if (logLag)
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[Lag] auto-lambda sweep %d/3 at %.6g\n", (int)iter + 1,
                       sweepLams[(int)iter]);
      } else {
        lambda = sweepHasUB ? sweepUBLam : sweepLBLam;
        sweepDone = true;
        if (logLag)
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[Lag] auto-lambda winner (%s %.6g), ascending\n",
                       sweepHasUB ? "UB" : "LB",
                       sweepHasUB ? sweepBestUB : sweepBestLB);
      }
    }
    // Modified costs for this multiplier vector.
    double lagLB = 0.0;
    bool allFresh = true;  // every block supplied a usable solution
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
      sublp.integrality_.resize(nbC);
      bool hasDiscrete = false;
      for (HighsInt j = 0; j != nbC; ++j) {
        sublp.integrality_[j] = model.integrality_[blk.cols[j]];
        if (sublp.integrality_[j] != HighsVarType::kContinuous)
          hasDiscrete = true;
      }
      if (!mipsolver.options_mip_->mip_lagrangian_subproblem_mip) {
        sublp.integrality_.assign(nbC, HighsVarType::kContinuous);
        hasDiscrete = false;
      }
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
      HighsSubLpResult res;
      if (hasDiscrete) {
        // Share this iteration's budget across blocks: with dozens of
        // blocks, per-block optimality is unaffordable, and capped
        // sub-MIPs still contribute VALID dual bounds below (optimality
        // is not required for the bound, only for the primal solution).
        // Parent gaps stop blocks once good enough for the parent.
        const HighsInt itersLeft = std::max<HighsInt>(1, maxIter - iter);
        const double loopLeft =
            (maxTime < kHighsInf)
                ? std::max(0.0, maxTime -
                                    (mipsolver.timer_.getWallTime() - lagStart))
                : kHighsInf;
        const double iterBudget =
            std::min(remaining, loopLeft) / (double)itersLeft;
        const HighsInt blocksLeft = nB - k;
        const double blockCap = std::max(
            0.02, std::min(2.0, iterBudget / std::max<HighsInt>(1, blocksLeft)));
        res = solveSubMip(sublp, std::min(blockCap, remaining),
                          mipsolver.options_mip_->mip_rel_gap,
                          mipsolver.options_mip_->mip_abs_gap);
      } else {
        res = solveSubLp(sublp, std::min(iterLpCap, remaining));
      }
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
      ++numAttempted;
      if (res.status == HighsModelStatus::kOptimal) {
        if ((HighsInt)res.colSol.size() != nbC) return true;
        blockSol[k] = res.colSol;
        if (!std::isfinite(res.obj)) return true;
        lagLB += res.obj;
        ++numOptimal;
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
      } else if (hasDiscrete && std::isfinite(res.dualBound)) {
        // Capped (non-optimal) MIP block: the solver's dual bound is a
        // valid lower bound on this block's minimum, so it contributes
        // to lagLB without proven block optimality. Proven optimality
        // was previously required only because no bound was harvested.
        // Validity rests solely on the harvested bound, never on the
        // multipliers or the step rule. Tripwire: dual > incumbent is
        // impossible and falls back instead of trusting anything.
        if ((HighsInt)res.colSol.size() == nbC) {
          if (res.dualBound >
              res.obj + 1e-6 * std::max(1.0, std::fabs(res.obj)) + 1e-9)
            return true;
          blockSol[k] = res.colSol;
        } else {
          allFresh = false;  // subgradient misses this block (step only)
        }
        lagLB += res.dualBound;
      } else {
        return true;  // unbounded subproblem or solver trouble: fallback
      }
    }
    // Auto-probe verdict (first iteration only): Lagrangian ascent needs
    // blocks that solve to proven optimality; mostly-timed-out blocks
    // mean the wrong shape, so abort instead of sweeping/ascenting.
    // Measured split: 100% optimal proceeds (Salihli), ~2% aborts
    // (Adana). Remembered across restarts.
    if (lagAuto && iter == 0 && numAttempted > 0 &&
        numOptimal * 5 < numAttempted * 4) {
      if (logLag)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Lag] auto probe: only %d/%d blocks optimal -> normal "
                     "MIP\n",
                     (int)numOptimal, (int)numAttempted);
      lagProbeFailed = true;
      return true;
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
    // Sweep bookkeeping: snapshot the multipliers behind the best
    // feasible composition (else best bound) for the ascent seed.
    if (!sweepDone) {
      if (hasUB && bestUB < sweepBestUB) {
        sweepBestUB = bestUB;
        sweepUBLam = lambda;
        sweepHasUB = true;
      }
      if (!sweepHasUB && bestLB > sweepBestLB) {
        sweepBestLB = bestLB;
        sweepLBLam = lambda;
      }
    }
    // Subgradient = coupling-row violations at the block solutions.
    // Blocks without a fresh solution reuse their last one for the step
    // direction (better than zero, still heuristic: bound validity never
    // depends on it), while allFresh gates the primal composition below
    // against stale mixing. Never-solved blocks are skipped by the size
    // check (their vectors are empty).
    std::vector<double> activity(numRow, 0.0);
    for (HighsInt k = 0; k != nB; ++k) {
      if ((HighsInt)blockSol[k].size() != (HighsInt)blocks[k].cols.size()) {
        allFresh = false;
        continue;
      }
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
    // Requires fresh solutions in every block (no stale mixing).
    bool couplingOk = allFresh;
    for (HighsInt a = 0; couplingOk && a != nA; ++a) {
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
                   "[Lag] iter %d: lit=%.6g best=%.6g UB=%.6g gnorm=%.3g\n",
                   (int)iter, lagLB, bestLB, bestUB, std::sqrt(gnorm2));
    const double gapTol = 1e-7 * std::max(1.0, std::fabs(bestUB));
    if (hasUB && bestUB - bestLB <= gapTol) {
      converged = true;
      break;
    }
    if (gnorm2 < 1e-24) break;  // stationary: gap decides fix vs fallback
    // Multiplier step: Polyak when a UB exists, else normalized
    // diminishing. (A scale-aware variant diverged via its C(|lambda|)
    // feedback loop on Salihli: lit 928 -> -1710 accelerating. The dual
    // needs a bundle method, not a step tweak; until then stay flat and
    // harmless. Documented, Salihli traj 2026-09-07.)
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

  // Parent-space bound for the log lines below (validity harness reads
  // them); the actual B&B injection is gated separately.
  const double parentLB = sign * bestLB + model.offset_;
  // Dual-bound injection, gated exactly like the Benders master bound:
  // weak bounds must never overwrite good ones (observed on Salihli: a
  // post-restart pass injecting -13751 over the tree's 3830, report-only
  // poison that makes proof-by-tolerance unreachable). Min-only (frame),
  // eps-shrunk, below-incumbent, strictly improving; direct assignment.
  if (numIter > 0 && std::isfinite(bestLB) &&
      model.sense_ == ObjSense::kMinimize) {
    const double parentLagLB = sign * bestLB + model.offset_;
    const double boundEps = 1e-7 * std::max(1.0, std::fabs(parentLagLB));
    const double injectLB = parentLagLB - boundEps;
    if (std::isfinite(injectLB) && injectLB <= upper_bound &&
        injectLB > lower_bound) {
      lower_bound = injectLB;
      if (logLag)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Lag] injected dual bound %.6g\n", injectLB);
    }
  }
  if (!hasUB) {
    if (logLag)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Lag] no feasible composition (%d iters, best dual "
                   "bound %.6g) -> normal MIP\n",
                   (int)numIter, parentLB);
    return true;
  }
  // Parent-space dual bound on every path with a finite bound (validity
  // harness + optional B&B injection read this line).
  if (logLag && std::isfinite(parentLB))
    highsLogUser(logOptions, HighsLogType::kInfo,
                 "[Lag] best dual bound %.6g (%d iters)\n", parentLB,
                 (int)numIter);
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

bool HighsMipSolverData::runLagRepair() {
  HighsLp& model = presolvedModel;
  const HighsInt numCol = model.num_col_;
  const HighsInt numRow = model.num_row_;
  if (numCol == 0 || numRow == 0) return true;
  if (numCol < 100) return true;
  if (!mipsolver.options_mip_->mip_decomposition) return true;
  // Detection serves the presolve repair below, the branching hint, and
  // the search-time ruin-and-recreate heuristic; it re-runs on every
  // presolve pass because restarts rebuild the model (dimensions may
  // shrink, which would strand a stored candidate). The expensive solves
  // only run on the initial pass.
  const bool runRepair = mipsolver.options_mip_->mip_lagrangian_repair;
  if (!runRepair &&
      !mipsolver.options_mip_->mip_heuristic_run_lagrepair)
    return true;
  if (model.a_matrix_.format_ != MatrixFormat::kColwise)
    model.a_matrix_.ensureColwise();
  const bool logRep = mipsolver.options_mip_->mip_decomposition_logging;
  const HighsLogOptions& logOptions = mipsolver.options_mip_->log_options;
  const double feastol = mipsolver.options_mip_->mip_feasibility_tolerance;
  const double maxTime = mipsolver.options_mip_->mip_lagrangian_repair_max_time;
  const HighsInt maxUnion = std::max<HighsInt>(
      1, mipsolver.options_mip_->mip_lagrangian_repair_max_cols);
  const double repStart = mipsolver.timer_.read();
  auto timeLeft = [&]() {
    double tl = kHighsInf;
    if (maxTime < kHighsInf)
      tl = maxTime - (mipsolver.timer_.read() - repStart);
    if (mipsolver.options_mip_->time_limit < kHighsInf)
      tl = std::min(
          tl, mipsolver.options_mip_->time_limit - mipsolver.timer_.read());
    return tl;
  };

  HighsLagCandidate cand;
  if (!findLagSeparator(model, cand)) {
    if (logRep)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[LagRepair] no candidate (%s) -> normal MIP\n",
                   cand.reason.c_str());
    return true;
  }
  // Share the decomposition with the search-time heuristic (validity is
  // rechecked at every use).
  lagRepairCand = cand;
  lagRepairCandValid = true;
  const double sepDone = mipsolver.timer_.read();
  const HighsInt nB = (HighsInt)cand.blockCols.size();
  const HighsInt maxBlocks = std::max<HighsInt>(
      2, mipsolver.options_mip_->mip_lagrangian_repair_max_blocks);
  if (nB > maxBlocks) {
    if (logRep)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[LagRepair] %d blocks (cap %d) -> normal MIP\n", (int)nB,
                   (int)maxBlocks);
    return true;
  }
  // Internal minimization (same single-convention pattern as Benders and
  // Lagrangian).
  const double sign = (model.sense_ == ObjSense::kMaximize) ? -1.0 : 1.0;

  std::vector<char> colFixed(numCol, 0);
  for (HighsInt c = 0; c != numCol; ++c) {
    if (model.col_lower_[c] == model.col_upper_[c]) {
      if (!std::isfinite(model.col_lower_[c])) return true;
      colFixed[c] = 1;
    }
  }
  // Branching priority (branch-on-bridges): publish unfixed columns
  // touching coupling rows into the shared coupling hint read by
  // structure-aware branching (OR-merged with any Benders separator
  // published earlier in this pass; cleared only by Benders entry and
  // convergence-and-fix). Fixing these columns disconnects the model,
  // so branching them first rediscovers the block structure in-tree.
  // Published before any solve so the hint exists even when every
  // subproblem below falls back. Initial pass only: post-restart models
  // are LP relaxations plus cuts, where the separator would be a cut
  // artifact.
  if (numRestarts == 0) {
    std::vector<char> inRc(numRow, 0);
    for (HighsInt r : cand.couplingRows) inRc[r] = 1;
    if ((HighsInt)bendersCoupling.size() != numCol)
      bendersCoupling.assign(numCol, 0);
    const HighsInt pubCap = 512;
    HighsInt numPub = 0;
    for (HighsInt c = 0; c != numCol && numPub < pubCap; ++c) {
      if (colFixed[c] || bendersCoupling[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el) {
        if (inRc[model.a_matrix_.index_[el]]) {
          bendersCoupling[c] = 1;
          ++numPub;
          break;
        }
      }
    }
    if (logRep)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[LagRepair] published %d coupling columns for branching "
                   "priority\n",
                   (int)numPub);
  }
  // Solves run on the initial pass with the presolve repair enabled;
  // detection above (candidate store, and the branching hint on the
  // initial pass) already ran for the search-time heuristic.
  if (numRestarts > 0 || !runRepair) return true;
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
  // Fully-determined rows must hold (same check as runLagrangian).
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

  // Directed coupling arcs (same frame as runLagrangian): free rows can
  // never be violated and carry no arc.
  struct RepArc {
    HighsInt row;
    HighsInt dir;  // +1: <= part, -1: >= part, 0: equality
    double bound;  // shifted U / L / b
  };
  std::vector<RepArc> arcs;
  for (HighsInt r : cand.couplingRows) {
    const double lo = model.row_lower_[r] == -kHighsInf
                          ? -kHighsInf
                          : model.row_lower_[r] - rowShift[r];
    const double hi = model.row_upper_[r] == kHighsInf
                          ? kHighsInf
                          : model.row_upper_[r] - rowShift[r];
    const bool hasLo = lo != -kHighsInf;
    const bool hasHi = hi != kHighsInf;
    if (!hasLo && !hasHi) continue;
    if (hasLo && hasHi && lo == hi) {
      arcs.push_back({r, 0, lo});
    } else {
      if (hasHi) arcs.push_back({r, +1, hi});
      if (hasLo) arcs.push_back({r, -1, lo});
    }
  }

  // Native MIP-start injection (same channel as runLagrangian: postsolve
  // to original space, re-check, publish only if feasible there).
  auto injectRepair = [&](const std::vector<double>& sol) -> bool {
    HighsSolution injsol;
    injsol.col_value = sol;
    injsol.value_valid = true;
    injsol.dual_valid = false;
    HighsBasis injbasis;
    injbasis.valid = false;
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
      if (logRep)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[LagRepair] injected incumbent (obj %.6g)\n",
                     double(injObj));
      return true;
    }
    if (logRep)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[LagRepair] postsolved incumbent infeasible (%.2g, %.2g, "
                   "%.2g) -> dropped\n",
                   boundViol, rowViol, intViol);
    return false;
  };

  // Standalone block subproblems (no multipliers: pure ruin step).
  // A usable solution only needs feasibility (primal UB purposes); only
  // a relaxation-proof infeasibility carries a global verdict.
  std::vector<std::vector<double>> blockSol(nB);
  std::vector<char> blockUsable(nB, 0);
  std::vector<char> blockFailed(nB, 0);
  HighsInt numFailed = 0;
  HighsInt numBlockOptimal = 0;
  // Internal-min-space sum of usable block objectives: a rough scale for
  // the joint result below ( orders of magnitude above it signal a
  // penalty corner, not a useful incumbent). Invalid when any block is
  // unsolved.
  double refInternal = 0.0;
  bool refValid = true;
  const double blocksStart = mipsolver.timer_.read();
  for (HighsInt k = 0; k != nB; ++k) {
    if (timeLeft() <= 0) return true;
    const std::vector<HighsInt>& cols = cand.blockCols[k];
    const std::vector<HighsInt>& rows = cand.blockRows[k];
    const HighsInt nbC = (HighsInt)cols.size();
    const HighsInt nbR = (HighsInt)rows.size();
    if (nbR == 0) {
      // Rowless block: analytic box minimum over internal-min costs.
      std::vector<double> sol(nbC, 0.0);
      bool bounded = true;
      for (HighsInt j = 0; j != nbC; ++j) {
        HighsInt c = cols[j];
        const double cj = sign * model.col_cost_[c];
        if (cj > 0) {
          if (!std::isfinite(model.col_lower_[c])) {
            bounded = false;
            break;
          }
          sol[j] = model.col_lower_[c];
        } else if (cj < 0) {
          if (!std::isfinite(model.col_upper_[c])) {
            bounded = false;
            break;
          }
          sol[j] = model.col_upper_[c];
        } else {
          sol[j] = std::isfinite(model.col_lower_[c])
                       ? model.col_lower_[c]
                       : 0.0;
        }
      }
      if (!bounded) return true;
      for (HighsInt j = 0; j != nbC; ++j) {
        const HighsInt c = cols[j];
        refInternal += (sign * model.col_cost_[c]) * sol[j];
      }
      blockSol[k] = std::move(sol);
      blockUsable[k] = 1;
      continue;
    }
    HighsLp sublp;
    sublp.num_col_ = nbC;
    sublp.num_row_ = nbR;
    sublp.sense_ = ObjSense::kMinimize;
    sublp.offset_ = 0.0;
    sublp.a_matrix_.format_ = MatrixFormat::kColwise;
    sublp.a_matrix_.start_.assign(nbC + 1, 0);
    sublp.col_cost_.resize(nbC);
    sublp.col_lower_.resize(nbC);
    sublp.col_upper_.resize(nbC);
    sublp.integrality_.resize(nbC);
    bool hasDiscrete = false;
    for (HighsInt j = 0; j != nbC; ++j) {
      HighsInt c = cols[j];
      sublp.col_cost_[j] = sign * model.col_cost_[c];
      sublp.col_lower_[j] = model.col_lower_[c];
      sublp.col_upper_[j] = model.col_upper_[c];
      sublp.integrality_[j] = model.integrality_[c];
      if (sublp.integrality_[j] != HighsVarType::kContinuous)
        hasDiscrete = true;
    }
    if (hasDiscrete && !mipsolver.options_mip_->mip_lagrangian_subproblem_mip)
      sublp.integrality_.assign(nbC, HighsVarType::kContinuous);
    sublp.row_lower_.resize(nbR);
    sublp.row_upper_.resize(nbR);
    for (HighsInt i = 0; i != nbR; ++i) {
      HighsInt r = rows[i];
      sublp.row_lower_[i] = model.row_lower_[r] == -kHighsInf
                                ? -kHighsInf
                                : model.row_lower_[r] - rowShift[r];
      sublp.row_upper_[i] = model.row_upper_[r] == kHighsInf
                                ? kHighsInf
                                : model.row_upper_[r] - rowShift[r];
    }
    std::vector<HighsInt> rowPos(numRow, -1);
    for (HighsInt i = 0; i != nbR; ++i) rowPos[rows[i]] = i;
    for (HighsInt j = 0; j != nbC; ++j) {
      HighsInt c = cols[j];
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el) {
        HighsInt sr = rowPos[model.a_matrix_.index_[el]];
        if (sr < 0) continue;  // separator-row entry (dropped in ruin step)
        sublp.a_matrix_.index_.push_back(sr);
        sublp.a_matrix_.value_.push_back(model.a_matrix_.value_[el]);
      }
      sublp.a_matrix_.start_[j + 1] = (HighsInt)sublp.a_matrix_.index_.size();
    }
    const double tl = timeLeft();
    // Parent gap tolerances: block solutions serve primal purposes only
    // (verified before any use), so incumbents good enough for the
    // parent stop the subsolver early instead of burning the budget.
    const double parentRelGap = mipsolver.options_mip_->mip_rel_gap;
    const double parentAbsGap = mipsolver.options_mip_->mip_abs_gap;
    HighsSubLpResult res =
        (hasDiscrete &&
         mipsolver.options_mip_->mip_lagrangian_subproblem_mip)
            ? solveSubMip(sublp, std::min(1.0, tl), parentRelGap,
                          parentAbsGap)
            : solveSubLp(sublp, std::min(10.0, tl));
    if (res.status == HighsModelStatus::kInfeasible) {
      // Block rows alone infeasible: the relaxation is infeasible, so the
      // whole model is infeasible.
      mipsolver.modelstatus_ = HighsModelStatus::kInfeasible;
      if (logRep)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[LagRepair] block %d infeasible without coupling -> "
                     "globally infeasible\n",
                     (int)k);
      return true;
    }
    // Usability (not optimality) decides: solved blocks compose, while
    // unsolved ones join the recreate union below instead of aborting.
    // No bound is ever derived here, so suboptimality is harmless.
    if ((HighsInt)res.colSol.size() != nbC) {
      blockFailed[k] = 1;
      ++numFailed;
      refValid = false;
      continue;
    }
    refInternal += res.obj;
    blockSol[k] = res.colSol;
    blockUsable[k] = 1;
    if (res.status == HighsModelStatus::kOptimal) ++numBlockOptimal;
  }
  if (logRep)
    highsLogUser(logOptions, HighsLogType::kInfo,
                 "[LagRepair] %d blocks solved (%d optimal, %d failed) in "
                 "%.1fs\n",
                 (int)nB, (int)numBlockOptimal, (int)numFailed,
                 mipsolver.timer_.read() - blocksStart);

  // Compose the ruin solution (fixed cols at bounds, solved blocks at
  // their solutions; unsolved blocks keep lower-bound placeholders and
  // always join the union below, so placeholders are never fixed).
  std::vector<double> composed(numCol, 0.0);
  for (HighsInt c = 0; c != numCol; ++c) composed[c] = model.col_lower_[c];
  for (HighsInt k = 0; k != nB; ++k) {
    if (!blockUsable[k]) continue;
    for (size_t j = 0; j != cand.blockCols[k].size(); ++j)
      composed[cand.blockCols[k][j]] = blockSol[k][j];
  }
  // Block membership for the recreate step.
  std::vector<HighsInt> blockOf(numCol, -1);
  for (HighsInt k = 0; k != nB; ++k)
    for (HighsInt c : cand.blockCols[k]) blockOf[c] = k;
  // Coupling violation at the composition (shifted frame: fixed activity
  // already removed from bounds). Per-block activities on coupling rows
  // double as ruin scores below.
  std::vector<HighsInt> rowCoupPos(numRow, -1);
  for (size_t i = 0; i != cand.couplingRows.size(); ++i)
    rowCoupPos[cand.couplingRows[i]] = (HighsInt)i;
  const HighsInt nRC = (HighsInt)cand.couplingRows.size();
  std::vector<double> activity(numRow, 0.0);
  std::vector<std::vector<double>> blockAct(nB,
                                            std::vector<double>(nRC, 0.0));
  for (HighsInt c = 0; c != numCol; ++c) {
    if (colFixed[c]) continue;
    const double v = composed[c];
    const HighsInt b = blockOf[c];
    for (HighsInt el = model.a_matrix_.start_[c];
         el != model.a_matrix_.start_[c + 1]; ++el) {
      const HighsInt r = model.a_matrix_.index_[el];
      const double add = model.a_matrix_.value_[el] * v;
      activity[r] += add;
      const HighsInt p = rowCoupPos[r];
      if (b >= 0 && p >= 0) blockAct[b][p] += add;
    }
  }
  auto arcViolated = [&](const RepArc& a) -> bool {
    const double tol = feastol * std::max(1.0, std::fabs(a.bound));
    const double act = activity[a.row];
    if (a.dir > 0) return act - a.bound > tol;
    if (a.dir < 0) return a.bound - act > tol;
    return std::fabs(act - a.bound) > tol;
  };
  std::vector<char> rowViolated(numRow, 0);
  bool anyViolated = false;
  for (const RepArc& a : arcs) {
    // One flag per row (ranged rows own two arcs).
    if (!rowViolated[a.row] && arcViolated(a)) {
      rowViolated[a.row] = 1;
      anyViolated = true;
    }
  }
  if (logRep)
    highsLogUser(logOptions, HighsLogType::kInfo,
                 "[LagRepair] candidate: %d coupling rows, %d blocks "
                 "(detect %.1fs); composition %s\n",
                 (int)cand.couplingRows.size(), (int)nB, sepDone - repStart,
                 anyViolated ? "violates coupling -> recreate"
                             : "coupling-feasible");
  if (!anyViolated && numFailed == 0) {
    // Nothing to recreate: offer the verified composition directly.
    // (With failed blocks the composition is incomplete, so the union
    // below always runs.)
    if (verifyBendersSolution(model, composed)) injectRepair(composed);
    return true;
  }
  // Recreate: union of blocks touching violated rows (row entries come
  // from the row-wise column lists built below); everything else stays
  // fixed at the composition. The union is a ranked prefix (failed blocks
  // first, then ruin scores) within the column cap, with one expansion to
  // twice the cap if the first joint proves infeasible. Cost is bounded
  // by the caps either way.
  // Row-wise column lists over unfixed columns for the union scan.
  std::vector<char> blockTouch(nB, 0);
  {
    std::vector<HighsInt> cnt(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el)
        ++cnt[model.a_matrix_.index_[el]];
    }
    std::vector<HighsInt> start(numRow + 1, 0);
    for (HighsInt r = 0; r != numRow; ++r) start[r + 1] = start[r] + cnt[r];
    std::vector<HighsInt> cols(start[numRow], -1);
    std::vector<HighsInt> fill(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el) {
        HighsInt r = model.a_matrix_.index_[el];
        cols[start[r] + fill[r]++] = c;
      }
    }
    for (HighsInt r = 0; r != numRow; ++r) {
      if (!rowViolated[r]) continue;
      for (HighsInt e = start[r]; e != start[r + 1]; ++e) {
        HighsInt b = blockOf[cols[e]];
        if (b >= 0) blockTouch[b] = 1;
      }
    }
  }
  // Unsolved blocks always join: their columns were never fixed, so
  // only the joint problem can place them.
  for (HighsInt k = 0; k != nB; ++k)
    if (blockFailed[k]) blockTouch[k] = 1;
  // Ranked order of touching blocks: unsolved blocks first (no scores
  // and no fixing values), then ruin scores (per-block activity in the
  // direction of the needed movement). Heuristic order only.
  std::vector<double> score(nB, 0.0);
  for (HighsInt i = 0; i != nRC; ++i) {
    HighsInt r = cand.couplingRows[i];
    if (!rowViolated[r]) continue;
    const double act = activity[r];
    const double lo = model.row_lower_[r] == -kHighsInf
                          ? -kHighsInf
                          : model.row_lower_[r] - rowShift[r];
    const double hi = model.row_upper_[r] == kHighsInf
                          ? kHighsInf
                          : model.row_upper_[r] - rowShift[r];
    const bool hasHi = hi != kHighsInf;
    // The row is violated, so act sits strictly outside [lo, hi]:
    // rank the side that must move (largest users first when the
    // activity must come down, smallest first when it must go up).
    const double dir = (hasHi && act > hi) ? 1.0 : -1.0;
    for (HighsInt k = 0; k != nB; ++k) score[k] += dir * blockAct[k][i];
  }
  std::vector<HighsInt> orderFailed;
  std::vector<HighsInt> order;
  for (HighsInt k = 0; k != nB; ++k) {
    if (!blockTouch[k]) continue;
    // Unsolved blocks have no scores and no fixing values: place them
    // first so a tight cap drops scored blocks instead of stranding
    // unfixable columns outside the union.
    if (blockFailed[k])
      orderFailed.push_back(k);
    else
      order.push_back(k);
  }
  std::stable_sort(order.begin(), order.end(), [&](HighsInt a, HighsInt b) {
    return score[a] > score[b];
  });
  std::vector<HighsInt> ranked;
  ranked.insert(ranked.end(), orderFailed.begin(), orderFailed.end());
  ranked.insert(ranked.end(), order.begin(), order.end());
  HighsInt numUnfixed = 0;
  for (HighsInt c = 0; c != numCol; ++c)
    if (!colFixed[c]) ++numUnfixed;
  // Up to two attempts: the ranked prefix within the column cap, then
  // (only on a proven-infeasible joint, where strictly more freedom is
  // the only thing that can help) one expansion to twice the cap.
  HighsInt cap = maxUnion;
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::vector<char> blockInU(nB, 0);
    std::vector<char> inU(numCol, 0);
    HighsInt numU = 0;
    HighsInt numFilled = 0;
    {
      HighsInt used = 0;
      for (HighsInt k : ranked) {
        const HighsInt sz = (HighsInt)cand.blockCols[k].size();
        if (used + sz > cap) continue;
        blockInU[k] = 1;
        used += sz;
        ++numFilled;
      }
    }
    for (HighsInt k = 0; k != nB; ++k) {
      if (!blockInU[k]) continue;
      for (HighsInt c : cand.blockCols[k]) {
        if (!inU[c]) {
          inU[c] = 1;
          ++numU;
        }
      }
    }
    if (numU == 0) {
      if (logRep)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[LagRepair] recreate union empty -> normal MIP\n");
      return true;
    }
    // A union holding more than the configured share of the unfixed
    // columns is a focused recreate only when the model itself is
    // large enough that branch-and-bound does not trivially own it;
    // on small models the same situation aborts (the parent search
    // owns that solve, and the joint budget would only delay it).
    const HighsInt maxUnionPct = std::max<HighsInt>(
        1, std::min<HighsInt>(
               100, mipsolver.options_mip_->mip_lagrangian_repair_max_union_pct));
    if (numUnfixed > 0 && numUnfixed <= 2 * maxUnion &&
        numU * 100 > maxUnionPct * numUnfixed) {
      if (logRep)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[LagRepair] recreate union %d of %d unfixed cols "
                     "(>%d%%, small model) -> normal MIP\n",
                     (int)numU, (int)numUnfixed, (int)maxUnionPct);
      return true;
    }
    HighsInt numUBlocks = 0;
    for (HighsInt k = 0; k != nB; ++k)
      if (blockInU[k]) ++numUBlocks;
    const double jointBudget = timeLeft();
    const double jointBudgetStart = mipsolver.timer_.read();
    if (jointBudget <= 0) return true;
    if (logRep)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[LagRepair] recreate union: %d cols from %d blocks, joint "
                   "budget %.1fs (attempt %d)\n",
                   (int)numU, (int)numUBlocks, jointBudget, attempt + 1);
    // Joint sub-MIP: the full presolved model with every column outside
    // the union fixed to the composition. A joint infeasibility proves
    // nothing globally (the fixing was our heuristic choice), so only a
    // verified composition is ever injected.
    HighsLp joint = model;
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c] || inU[c]) continue;
      joint.col_lower_[c] = joint.col_upper_[c] = composed[c];
    }
    HighsSubMipProgress progress;
    // Auto-exit tripwires: good-enough needs no option (abort when the
    // incumbent is within parent gap tolerance of the dual bound; dead
    // when no dual bound exists yet). Stagnation is configured. Both
    // interrupt mid-solve; banked incumbents harvest as usual.
    if (model.sense_ == ObjSense::kMinimize &&
        std::isfinite(lower_bound)) {
      const double rTol = mipsolver.options_mip_->mip_rel_gap;
      const double aTol = mipsolver.options_mip_->mip_abs_gap;
      progress.targetBound =
          lower_bound + std::max(aTol, rTol * std::fabs(lower_bound));
    }
    progress.minStallNodes = std::max<int64_t>(
        0, mipsolver.options_mip_->mip_lagrangian_repair_stall_nodes);
    progress.stallSeconds =
        mipsolver.options_mip_->mip_lagrangian_repair_stall_seconds;
    HighsSubLpResult res =
        solveSubMip(joint, timeLeft(), mipsolver.options_mip_->mip_rel_gap,
                    mipsolver.options_mip_->mip_abs_gap, &progress);
    const double jointDone = mipsolver.timer_.read();
    if (logRep) {
      std::lock_guard<std::mutex> guard(progress.mutex);
      for (const auto& e : progress.events)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[LagRepair-joint] incumbent %.6g at %.1fs\n", e.second,
                     e.first);
    }
    if (logRep)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[LagRepair] joint: status=%d obj=%.6g solsize=%d/%d "
                   "(solve %.1fs, repair total %.1fs)\n",
                   (int)res.status, res.obj, (int)res.colSol.size(),
                   (int)numCol, jointDone - jointBudgetStart,
                   jointDone - repStart);
    // Interrupted joints (auto-exit tripwire or external) still harvest
    // whatever the callback banked; the gates below decide.
    if (logRep && res.status == HighsModelStatus::kInterrupt)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[LagRepair] joint interrupted -> harvested\n");
    if ((HighsInt)res.colSol.size() == numCol) {
      if (verifyBendersSolution(model, res.colSol)) {
        // Scale sanity: the joint result must sit within an order of
        // magnitude of the block-objective sum (both in parent space);
        // far above it signals a penalty corner produced by a wrong
        // union, not a useful incumbent. Calibrated: legitimate coupling
        // cost measured 1.0x (Adana) and 17.5x (synthetic), versus 50x+
        // for penalty corners. Validity is unaffected (verify passed);
        // only the injection is skipped.
        bool sane = true;
        if (refValid && std::isfinite(refInternal)) {
          const double refParent = sign * refInternal + model.offset_;
          if (std::isfinite(refParent) &&
              res.obj > refParent + 30.0 * std::max(1.0, std::fabs(refParent)))
            sane = false;
        }
        if (sane) {
          injectRepair(res.colSol);
        } else if (logRep) {
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[LagRepair] joint obj %.6g far above block-ref %.6g "
                       "-> dropped\n",
                       res.obj, sign * refInternal + model.offset_);
        }
        return true;
      }
      if (logRep) {
        double maxBnd = 0.0, maxRow = 0.0, maxInt = 0.0;
        for (HighsInt c = 0; c != numCol; ++c) {
          const double v = res.colSol[c];
          if (v < model.col_lower_[c])
            maxBnd = std::max(maxBnd, model.col_lower_[c] - v);
          if (v > model.col_upper_[c])
            maxBnd = std::max(maxBnd, v - model.col_upper_[c]);
          const HighsVarType it = model.integrality_[c];
          if (it == HighsVarType::kInteger ||
              it == HighsVarType::kSemiInteger ||
              it == HighsVarType::kImplicitInteger)
            maxInt = std::max(maxInt, std::fabs(v - std::round(v)));
        }
        if (model.a_matrix_.format_ == MatrixFormat::kColwise) {
          std::vector<double> act(numRow, 0.0);
          for (HighsInt c = 0; c != numCol; ++c) {
            for (HighsInt el = model.a_matrix_.start_[c];
                 el != model.a_matrix_.start_[c + 1]; ++el)
              act[model.a_matrix_.index_[el]] +=
                  model.a_matrix_.value_[el] * res.colSol[c];
          }
          for (HighsInt r = 0; r != numRow; ++r) {
            if (act[r] < model.row_lower_[r])
              maxRow = std::max(maxRow, model.row_lower_[r] - act[r]);
            if (act[r] > model.row_upper_[r])
              maxRow = std::max(maxRow, act[r] - model.row_upper_[r]);
          }
        }
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[LagRepair] joint solution failed verification "
                     "(maxbnd=%.2g maxrow=%.2g maxint=%.2g, feastol=%.2g) -> "
                     "normal MIP\n",
                     maxBnd, maxRow, maxInt, feastol);
      }
      return true;
    }
    if (logRep)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[LagRepair] joint status %d -> %s\n", (int)res.status,
                   res.status == HighsModelStatus::kInfeasible &&
                           numFilled < (HighsInt)ranked.size()
                       ? "expanding union"
                       : "normal MIP");
    if (res.status != HighsModelStatus::kInfeasible) return true;
    if (numFilled >= (HighsInt)ranked.size()) return true;
    cap = 2 * maxUnion;
  }
  return true;
}

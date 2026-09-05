/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// Classical Benders decomposition for weakly-coupled MIPs (clean-room
// implementation of the textbook method; no third-party code).
//
// Scope: arrowhead structures with a small column separator S whose
// removal leaves blocks coupled only through S. LP blocks (all
// continuous) give exact dual cuts. Blocks with discrete columns are
// supported via LP-relaxation cuts (valid but weaker), sub-MIP upper
// bounds and binary no-good cuts; see
// mip_benders_integer_subproblems. Anything else falls back.
// The master problem (mixed-integer over S plus one surrogate theta per
// block) is solved with the in-process exact MIP subsolver; LP
// subproblems are presolve-off simplex solves so duals/rays refer
// directly to the passed submodel.
//
// Correctness contract (mirrors the components path):
// - cuts are only added after a tightness/violation gate at the
//   generating point; any gate failure aborts Benders silently and the
//   normal MIP path continues with nothing fixed;
// - when an infeasible block yields no usable Farkas ray, an
//   always-feasible auxiliary LP (artificials relaxing the shifted block
//   rows) is solved and a feasibility cut is built from its optimal
//   duals instead; auxiliary failure also falls back silently;
// - the final composed solution is verified against the full presolved
//   model before any coupling column is fixed; a verified composition
//   is also injected as a native MIP-start incumbent when the loop does
//   not converge (rescue), so fallback never discards a feasible UB;
// - fixing is via bounds (dimensions preserved, postsolve consistent),
//   after which the parent MIP continues and produces its own proof.

#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Highs.h"
#include "mip/HighsMipSolver.h"

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

// Row adjacency over non-skipped columns (shared by auto-detection and
// the .dec annotation path): rowStart/rowCols list, per row, the kept
// columns touching it.
static void buildRowAdjacency(const HighsLp& model,
                              const std::vector<char>& skip,
                              std::vector<HighsInt>& rowStart,
                              std::vector<HighsInt>& rowCols) {
  const HighsInt numCol = model.num_col_;
  const HighsInt numRow = model.num_row_;
  rowStart.assign(numRow + 1, 0);
  {
    std::vector<HighsInt> cnt(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (skip[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el)
        ++cnt[model.a_matrix_.index_[el]];
    }
    for (HighsInt r = 0; r != numRow; ++r)
      rowStart[r + 1] = rowStart[r] + cnt[r];
  }
  rowCols.assign(rowStart[numRow], -1);
  {
    std::vector<HighsInt> fill(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (skip[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el) {
        HighsInt r = model.a_matrix_.index_[el];
        rowCols[rowStart[r] + fill[r]++] = c;
      }
    }
  }
}

}  // namespace

// Shared presolve-off simplex LP solve (also used by Lagrangian
// subproblems); declared on HighsMipSolverData for cross-TU use.
HighsMipSolverData::HighsSubLpResult HighsMipSolverData::solveSubLp(
    const HighsLp& sublp, double timeLimit) {
  HighsSubLpResult res;
  Highs lpsolver;
  lpsolver.setOptionValue("output_flag", false);
  // No presolve: duals/rays must refer to exactly the passed submodel.
  // No IPM: only simplex furnishes bases and Farkas rays.
  lpsolver.setOptionValue("presolve", kHighsOffString);
  lpsolver.setOptionValue("solver", kSimplexString);
  lpsolver.setOptionValue("time_limit", timeLimit);
  if (lpsolver.passModel(sublp) != HighsStatus::kOk) return res;
  if (lpsolver.run() != HighsStatus::kOk) return res;
  res.status = lpsolver.getModelStatus();
  if (res.status == HighsModelStatus::kOptimal) {
    const HighsSolution& sol = lpsolver.getSolution();
    res.dualValid = sol.dual_valid;
    res.colSol = sol.col_value;
    res.rowDual = sol.row_dual;
    res.obj = lpsolver.getInfo().objective_function_value;
  }
  return res;
}

// Annotation file format (mip_benders_dec_file), SCIP-.dec-inspired but
// HiGHS-local (clean-room): '#' comments, case-insensitive keywords,
//   COUPLING: <tokens...>
//   BLOCK: <tokens...>   (optional, repeatable)
// Tokens are original-model column indices (integers) or presolved
// column names. Fixed columns in either section are dropped silently
// (constants need no separator); anything unresolvable aborts the file
// and auto-detection runs instead. Without BLOCK lines the remainder is
// auto-partitioned; with BLOCK lines they must cover every unfixed
// non-separator column exactly once.
bool HighsMipSolverData::tryBendersDecFile(
    const HighsLp& model, HighsBendersCandidate& cand) const {
  cand = HighsBendersCandidate();
  const std::string& path = mipsolver.options_mip_->mip_benders_dec_file;
  if (path.empty() || path == kHighsFilenameDefault) return false;
  const bool logBend = mipsolver.options_mip_->mip_decomposition_logging;
  const HighsLogOptions& logOptions = mipsolver.options_mip_->log_options;
  auto fail = [&](const std::string& why) -> bool {
    cand = HighsBendersCandidate();
    cand.reason = "dec " + why;
    if (logBend)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Benders] dec file ignored (%s) -> auto-detect\n",
                   why.c_str());
    return false;
  };
  if (model.a_matrix_.format_ != MatrixFormat::kColwise)
    return fail("matrix not colwise");
  const HighsInt numCol = model.num_col_;
  const HighsInt numRow = model.num_row_;
  if (numCol <= 0 || numRow <= 0) return fail("empty model");
  std::ifstream file(path);
  if (!file.is_open()) return fail("unreadable");
  std::vector<std::string> couplingTok;
  std::vector<std::vector<std::string>> blockToks;
  std::string line;
  while (std::getline(file, line)) {
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    std::istringstream ls(line);
    std::string kw;
    if (!(ls >> kw)) continue;
    for (char& ch : kw) ch = (char)std::toupper((unsigned char)ch);
    if (!kw.empty() && kw.back() == ':') kw.pop_back();
    std::vector<std::string> toks;
    std::string tok;
    while (ls >> tok) toks.push_back(tok);
    if (kw == "COUPLING") {
      if (!couplingTok.empty()) return fail("duplicate COUPLING");
      couplingTok = std::move(toks);
    } else if (kw == "BLOCK") {
      if (toks.empty()) return fail("empty BLOCK");
      blockToks.push_back(std::move(toks));
    } else {
      return fail("unknown keyword '" + kw + "'");
    }
  }
  if (couplingTok.empty()) return fail("no COUPLING");
  // Original->presolved inverse map via the postsolve stack (always
  // initialized by presolve before this hook runs, same reliance as the
  // Lagrangian postsolve). Original size comes authoritatively from the
  // original model.
  const HighsLp* origLp = mipsolver.orig_model_;
  if (origLp == nullptr || origLp->num_col_ <= 0) return fail("no orig model");
  const HighsInt origNumCol = origLp->num_col_;
  std::vector<HighsInt> origToPres(origNumCol, -1);
  for (HighsInt c = 0; c != numCol; ++c) {
    const HighsInt o = postSolveStack.getOrigColIndex(c);
    if (o >= 0 && o < origNumCol) origToPres[o] = c;
  }
  // Presolved-name map (unique names only).
  std::unordered_map<std::string, HighsInt> nameToCol;
  std::unordered_map<std::string, HighsInt> nameCount;
  if ((HighsInt)model.col_names_.size() == numCol) {
    for (HighsInt c = 0; c != numCol; ++c) {
      nameToCol[model.col_names_[c]] = c;
      ++nameCount[model.col_names_[c]];
    }
  }
  const HighsLp* orig = origLp;
  // Resolve one token to a presolved column. Returns false on hard
  // failure; droppedFixed reports a presolve-removed fixed column.
  auto resolve = [&](const std::string& tok, HighsInt& presCol,
                     bool& droppedFixed) -> bool {
    droppedFixed = false;
    bool isInt = !tok.empty() && (std::isdigit((unsigned char)tok[0]) ||
                                  tok[0] == '-' || tok[0] == '+');
    for (size_t i = 1; isInt && i < tok.size(); ++i)
      if (!std::isdigit((unsigned char)tok[i])) isInt = false;
    if (isInt) {
      const long o = std::strtol(tok.c_str(), nullptr, 10);
      if (o < 0 || o >= origNumCol) return false;
      const HighsInt p = origToPres[(HighsInt)o];
      if (p < 0) {
        // Removed by presolve: acceptable only if fixed originally.
        if (orig == nullptr ||
            (HighsInt)orig->col_lower_.size() <= (HighsInt)o)
          return false;
        if (orig->col_lower_[o] != orig->col_upper_[o]) return false;
        droppedFixed = true;
        return true;
      }
      presCol = p;
      return true;
    }
    const auto itc = nameCount.find(tok);
    if (itc == nameCount.end() || itc->second != 1) return false;
    presCol = nameToCol[tok];
    return true;
  };
  std::vector<char> colFixed(numCol, 0);
  for (HighsInt c = 0; c != numCol; ++c) {
    if (model.col_lower_[c] == model.col_upper_[c]) {
      if (!std::isfinite(model.col_lower_[c])) return fail("degenerate");
      colFixed[c] = 1;
    }
  }
  std::vector<char> inS(numCol, 0);
  for (const std::string& tok : couplingTok) {
    HighsInt p = -1;
    bool dropped = false;
    if (!resolve(tok, p, dropped))
      return fail("bad COUPLING token '" + tok + "'");
    if (dropped || colFixed[p]) continue;
    inS[p] = 1;
  }
  HighsInt numS = 0;
  for (HighsInt c = 0; c != numCol; ++c) numS += inS[c] ? 1 : 0;
  if (numS == 0) return fail("empty separator");
  const HighsInt maxCoupling = std::max<HighsInt>(
      1, mipsolver.options_mip_->mip_benders_max_coupling_cols);
  if (numS > maxCoupling) return fail("coupling budget");
  std::vector<std::vector<HighsInt>> pieces;
  if (!blockToks.empty()) {
    std::vector<char> used(numCol, 0);
    for (const auto& bt : blockToks) {
      pieces.emplace_back();
      for (const std::string& tok : bt) {
        HighsInt p = -1;
        bool dropped = false;
        if (!resolve(tok, p, dropped))
          return fail("bad BLOCK token '" + tok + "'");
        if (dropped || colFixed[p]) continue;
        if (inS[p] || used[p]) return fail("BLOCK overlap");
        used[p] = 1;
        pieces.back().push_back(p);
      }
      if (pieces.back().empty()) return fail("empty BLOCK after mapping");
    }
    // Explicit blocks must cover every unfixed non-separator column.
    for (HighsInt c = 0; c != numCol; ++c) {
      if (!colFixed[c] && !inS[c] && !used[c])
        return fail("BLOCK lists incomplete");
    }
  } else {
    // Auto-partition the remainder (same DSU piece step as detection).
    std::vector<HighsInt> rowStart, rowCols;
    buildRowAdjacency(model, colFixed, rowStart, rowCols);
    DsU dsu(numCol);
    for (HighsInt r = 0; r != numRow; ++r) {
      HighsInt first = -1;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        HighsInt c = rowCols[e];
        if (inS[c]) continue;
        if (first < 0)
          first = c;
        else
          dsu.unite(first, c);
      }
    }
    std::vector<HighsInt> rootToPiece(numCol, -1);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c] || inS[c]) continue;
      HighsInt root = dsu.find(c);
      if (rootToPiece[root] < 0) {
        rootToPiece[root] = (HighsInt)pieces.size();
        pieces.emplace_back();
      }
      pieces[rootToPiece[root]].push_back(c);
    }
  }
  if ((HighsInt)pieces.size() < 2) return fail("fewer than two blocks");
  // Block kinds: same rule as auto-detection (LP blocks give exact dual
  // cuts; discrete blocks need integer-subproblem support).
  const bool allowMipBlocks =
      mipsolver.options_mip_->mip_benders_integer_subproblems;
  cand.blockIsLp.assign(pieces.size(), 1);
  for (size_t i = 0; i != pieces.size(); ++i) {
    for (HighsInt c : pieces[i]) {
      if (model.integrality_[c] != HighsVarType::kContinuous) {
        if (!allowMipBlocks) return fail("integer blocks unsupported");
        cand.blockIsLp[i] = 0;
        break;
      }
    }
  }
  // Row assignment with the defensive span-check (same as detection: a
  // row touching two blocks invalidates the annotation).
  std::vector<HighsInt> rowStart, rowCols;
  buildRowAdjacency(model, colFixed, rowStart, rowCols);
  std::vector<HighsInt> blockOf(numCol, -1);
  for (size_t k = 0; k != pieces.size(); ++k)
    for (HighsInt c : pieces[k]) blockOf[c] = (HighsInt)k;
  for (HighsInt c = 0; c != numCol; ++c)
    if (inS[c]) cand.couplingCols.push_back(c);
  cand.blockCols = pieces;
  cand.blockRows.assign(pieces.size(), {});
  std::vector<HighsInt> seenBlock;
  seenBlock.reserve(8);
  for (HighsInt r = 0; r != numRow; ++r) {
    seenBlock.clear();
    for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
      HighsInt c = rowCols[e];
      if (inS[c]) continue;
      HighsInt b = blockOf[c];
      if (b < 0) {
        // Unfixed column in no block: only possible with explicit
        // BLOCK lists, already checked complete above; defensive abort.
        cand = HighsBendersCandidate();
        cand.reason = "dec orphan column";
        return false;
      }
      if (std::find(seenBlock.begin(), seenBlock.end(), b) ==
          seenBlock.end())
        seenBlock.push_back(b);
    }
    if (seenBlock.empty()) {
      cand.masterRows.push_back(r);
    } else if (seenBlock.size() == 1) {
      cand.blockRows[seenBlock[0]].push_back(r);
    } else {
      cand = HighsBendersCandidate();
      cand.reason = "dec row spans two blocks";
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] dec file row %d spans blocks -> auto-detect\n",
                     (int)r);
      return false;
    }
  }
  cand.valid = true;
  cand.reason = "dec file";
  if (logBend)
    highsLogUser(logOptions, HighsLogType::kInfo,
                 "[Benders] candidate from dec file: %d coupling cols, %d "
                 "blocks\n",
                 (int)cand.couplingCols.size(), (int)cand.blockCols.size());
  return true;
}

bool HighsMipSolverData::findBendersSeparator(
    const HighsLp& model, HighsBendersCandidate& cand) const {
  cand = HighsBendersCandidate();
  // Explicit user annotation first (falls back to auto-detection on any
  // failure); explicit structure wins when valid.
  if (tryBendersDecFile(model, cand)) return true;
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
  std::vector<HighsInt> degree(numCol, 0);
  for (HighsInt c = 0; c != numCol; ++c)
    degree[c] = model.a_matrix_.start_[c + 1] - model.a_matrix_.start_[c];

  const HighsInt maxCoupling = std::max<HighsInt>(
      1, mipsolver.options_mip_->mip_benders_max_coupling_cols);
  const HighsInt minBlock = std::max<HighsInt>(
      1, mipsolver.options_mip_->mip_benders_min_block_cols);

  // Row adjacency over unfixed columns (built once; separator columns
  // are skipped during the scans).
  std::vector<HighsInt> rowStart;
  std::vector<HighsInt> rowCols;
  buildRowAdjacency(model, colFixed, rowStart, rowCols);

  // Separator search: repeatedly split the largest remaining piece with
  // the best verified single-column cut. A cut is only accepted if no row
  // still spans the resulting pieces (a spanning row would keep coupling
  // the blocks even with the column fixed). Deterministic (index
  // tie-breaks); any search failure only rejects candidacy, never harms
  // correctness (the row assignment below re-verifies independently).
  HighsInt totalNnz = 0;
  for (HighsInt c = 0; c != numCol; ++c)
    if (!colFixed[c]) totalNnz += degree[c];
  // Candidate cap scales with model size (heuristic): the scan is
  // O(cap * nnz) per round.
  const HighsInt scanCap = std::max<HighsInt>(
      25, std::min<HighsInt>(500, 2000000 / std::max<HighsInt>(1, totalNnz)));
  std::vector<char> inS(numCol, 0);
  HighsInt numS = 0;
  // Pieces of the graph without S.
  auto computePieces = [&](std::vector<std::vector<HighsInt>>& pieces) {
    DsU dsu(numCol);
    for (HighsInt r = 0; r != numRow; ++r) {
      HighsInt first = -1;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        HighsInt c = rowCols[e];
        if (inS[c]) continue;
        if (first < 0)
          first = c;
        else
          dsu.unite(first, c);
      }
    }
    pieces.clear();
    std::vector<HighsInt> rootToPiece(numCol, -1);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c] || inS[c]) continue;
      HighsInt root = dsu.find(c);
      if (rootToPiece[root] < 0) {
        rootToPiece[root] = (HighsInt)pieces.size();
        pieces.emplace_back();
      }
      pieces[rootToPiece[root]].push_back(c);
    }
  };
  // Nontrivial-piece count after additionally removing candidate col.
  auto splitCount = [&](HighsInt excl) -> HighsInt {
    DsU dsu(numCol);
    for (HighsInt r = 0; r != numRow; ++r) {
      HighsInt first = -1;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        HighsInt c = rowCols[e];
        if (c == excl || inS[c]) continue;
        if (first < 0)
          first = c;
        else
          dsu.unite(first, c);
      }
    }
    std::vector<HighsInt> sizes(numCol, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (c == excl || colFixed[c] || inS[c]) continue;
      ++sizes[dsu.find(c)];
    }
    HighsInt nontrivial = 0;
    for (HighsInt s : sizes)
      if (s >= minBlock) ++nontrivial;
    return nontrivial;
  };
  std::vector<std::vector<HighsInt>> searchPieces;
  for (;;) {
    computePieces(searchPieces);
    HighsInt curCount = 0;
    HighsInt largest = 0;
    HighsInt largestIdx = -1;
    for (size_t i = 0; i != searchPieces.size(); ++i) {
      const HighsInt sz = (HighsInt)searchPieces[i].size();
      if (sz >= minBlock) ++curCount;
      if (sz > largest) {
        largest = sz;
        largestIdx = (HighsInt)i;
      }
    }
    if (largestIdx < 0 || largest < 2 * minBlock) break;
    if (numS >= maxCoupling) break;  // finalization decides validity
    // Rank candidates inside the largest piece (degree filter prunes
    // leaves, which can never disconnect anything).
    HighsInt scanned = 0;
    HighsInt topC[3] = {-1, -1, -1};
    HighsInt topN[3] = {-1, -1, -1};
    for (HighsInt c : searchPieces[largestIdx]) {
      if (degree[c] < 2 || degree[c] > 32) continue;
      if (scanned >= scanCap) break;
      ++scanned;
      const HighsInt q = splitCount(c);
      for (HighsInt t = 0; t != 3; ++t) {
        if (q > topN[t]) {
          for (HighsInt u = 2; u != t; --u) {
            topN[u] = topN[u - 1];
            topC[u] = topC[u - 1];
          }
          topN[t] = q;
          topC[t] = c;
          break;
        }
      }
    }
    // Accept the first strict improvement, or a purification move: a
    // discrete column whose removal keeps the piece count but takes
    // discretes out of blocks (the master handles them exactly; leaving
    // them would force weaker MIP blocks). Continuous non-improving
    // moves stay rejected. S strictly grows, so this terminates at the
    // coupling budget. (Every remaining row touches exactly one DSU
    // piece by construction, so counting pieces is already a sound
    // separator test; the row assignment below re-verifies defensively.)
    bool accepted = false;
    for (HighsInt t = 0; t != 3; ++t) {
      if (topC[t] < 0) break;
      const HighsInt q = splitCount(topC[t]);
      const bool discrete =
          model.integrality_[topC[t]] != HighsVarType::kContinuous;
      if (q >= 2 &&
          (q > curCount || (q == curCount && discrete && curCount >= 2))) {
        inS[topC[t]] = 1;
        ++numS;
        accepted = true;
        if (mipsolver.options_mip_->mip_decomposition_logging)
          highsLogUser(mipsolver.options_mip_->log_options,
                       HighsLogType::kInfo,
                       "[Benders] separator: add column %d (pieces %d -> %d)\n",
                       (int)topC[t], (int)curCount, (int)q);
        break;
      }
    }
    if (!accepted) break;
  }

  // Finalize the separator to a fixpoint. Tiny or rowless pieces are
  // always absorbed into S. Discrete columns prefer S as well (the
  // master handles them exactly, keeping subproblems LP); only the
  // overflow beyond the coupling budget stays in MIP blocks, and only
  // when integer-subproblem support is enabled. Adding columns to S
  // cannot invalidate separation; the row assignment below re-verifies.
  const bool allowMipBlocks =
      mipsolver.options_mip_->mip_benders_integer_subproblems;
  for (;;) {
    DsU dsu(numCol);
    for (HighsInt r = 0; r != numRow; ++r) {
      HighsInt first = -1;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        HighsInt c = rowCols[e];
        if (inS[c]) continue;
        if (first < 0)
          first = c;
        else
          dsu.unite(first, c);
      }
    }
    std::vector<HighsInt> pieceSize(numCol, 0);
    std::vector<char> pieceHasRow(numCol, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c] || inS[c]) continue;
      HighsInt root = dsu.find(c);
      ++pieceSize[root];
      if (degree[c] > 0) pieceHasRow[root] = 1;
    }
    bool changed = false;
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c] || inS[c]) continue;
      HighsInt root = dsu.find(c);
      if (pieceSize[root] < minBlock || !pieceHasRow[root]) {
        inS[c] = 1;
        ++numS;
        changed = true;
      }
    }
    if (changed) {
      if (numS > maxCoupling) {
        cand.reason = "separator exceeds coupling budget after merge";
        return false;
      }
      continue;
    }
    // Kept pieces only from here on.
    std::vector<HighsInt> discreteCols;
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c] || inS[c]) continue;
      if (model.integrality_[c] != HighsVarType::kContinuous)
        discreteCols.push_back(c);
    }
    if (discreteCols.empty()) break;
    if (numS + (HighsInt)discreteCols.size() <= maxCoupling) {
      for (HighsInt c : discreteCols) {
        inS[c] = 1;
        ++numS;
      }
      continue;
    }
    if (allowMipBlocks) break;  // overflow stays in MIP blocks
    cand.reason = "integer blocks unsupported";
    return false;
  }
  // Collect kept blocks (deterministic order by first column).
  std::vector<std::vector<HighsInt>> pieces;
  {
    DsU dsu(numCol);
    for (HighsInt r = 0; r != numRow; ++r) {
      HighsInt first = -1;
      for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
        HighsInt c = rowCols[e];
        if (inS[c]) continue;
        if (first < 0)
          first = c;
        else
          dsu.unite(first, c);
      }
    }
    std::vector<HighsInt> rootToPiece(numCol, -1);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (colFixed[c] || inS[c]) continue;
      HighsInt root = dsu.find(c);
      if (rootToPiece[root] < 0) {
        rootToPiece[root] = (HighsInt)pieces.size();
        pieces.emplace_back();
      }
      pieces[rootToPiece[root]].push_back(c);
    }
  }
  if ((HighsInt)pieces.size() < 2) {
    cand.reason = "fewer than two blocks";
    if (mipsolver.options_mip_->mip_decomposition_logging) {
      std::string dbg;
      char buf[64];
      snprintf(buf, sizeof(buf), " |S|=%d piecesizes=", (int)numS);
      dbg += buf;
      std::vector<HighsInt> szs;
      // Recompute all DSU piece sizes (including absorbed) for diagnosis.
      DsU dsu2(numCol);
      for (HighsInt r = 0; r != numRow; ++r) {
        HighsInt first = -1;
        for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
          HighsInt c = rowCols[e];
          if (first < 0)
            first = c;
          else
            dsu2.unite(first, c);
        }
      }
      std::vector<HighsInt> allSizes(numCol, 0);
      for (HighsInt c = 0; c != numCol; ++c) {
        if (colFixed[c]) continue;
        ++allSizes[dsu2.find(c)];
      }
      for (HighsInt s : allSizes)
        if (s > 0) szs.push_back(s);
      std::sort(szs.begin(), szs.end());
      for (HighsInt s : szs) {
        snprintf(buf, sizeof(buf), "%d,", (int)s);
        dbg += buf;
      }
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
                   "[Benders] diagnose%s\n", dbg.c_str());
    }
    return false;
  }
  // Block kinds: LP blocks (all continuous) give exact dual cuts;
  // blocks with discrete columns need integer-subproblem support. If
  // that support is disabled, any discrete block rejects the candidate.
  cand.blockIsLp.assign(pieces.size(), 1);
  if (mipsolver.options_mip_->mip_benders_integer_subproblems) {
    for (size_t i = 0; i != pieces.size(); ++i) {
      for (HighsInt c : pieces[i]) {
        if (model.integrality_[c] != HighsVarType::kContinuous) {
          cand.blockIsLp[i] = 0;
          break;
        }
      }
    }
  } else {
    // LP blocks only (defensive: the fixpoint above already moved every
    // discrete column into S).
    for (const auto& piece : pieces) {
      for (HighsInt c : piece) {
        if (model.integrality_[c] != HighsVarType::kContinuous) {
          cand.reason = "block with discrete columns";
          return false;
        }
      }
    }
  }
  for (HighsInt c = 0; c != numCol; ++c)
    if (inS[c]) cand.couplingCols.push_back(c);
  cand.blockCols = std::move(pieces);
  // Assign rows: rows touching no block are master rows; rows touching
  // exactly one block are block rows. Anything else is a separator bug:
  // reject instead of risking an invalid decomposition.
  std::vector<HighsInt> blockOf(numCol, -1);
  for (size_t k = 0; k != cand.blockCols.size(); ++k)
    for (HighsInt c : cand.blockCols[k]) blockOf[c] = (HighsInt)k;
  cand.blockRows.assign(cand.blockCols.size(), {});
  std::vector<HighsInt> seenBlock;
  seenBlock.reserve(8);
  for (HighsInt r = 0; r != numRow; ++r) {
    seenBlock.clear();
    for (HighsInt e = rowStart[r]; e != rowStart[r + 1]; ++e) {
      HighsInt c = rowCols[e];
      if (inS[c]) continue;
      HighsInt b = blockOf[c];
      if (b < 0) continue;  // fixed column (not in rowCols by construction)
      if (std::find(seenBlock.begin(), seenBlock.end(), b) ==
          seenBlock.end())
        seenBlock.push_back(b);
    }
    if (seenBlock.empty()) {
      cand.masterRows.push_back(r);
    } else if (seenBlock.size() == 1) {
      cand.blockRows[seenBlock[0]].push_back(r);
    } else {
      cand = HighsBendersCandidate();
      cand.reason = "row spans two blocks (separator bug)";
      return false;
    }
  }
  cand.valid = true;
  cand.reason = "ok";
  return true;
}

bool HighsMipSolverData::verifyBendersSolution(
    const HighsLp& model, const std::vector<double>& sol) const {
  const double tol = mipsolver.options_mip_->mip_feasibility_tolerance;
  if (sol.size() != (size_t)model.num_col_) return false;
  if (model.a_matrix_.format_ != MatrixFormat::kColwise) return false;
  for (HighsInt c = 0; c != model.num_col_; ++c) {
    const double v = sol[c];
    if (!std::isfinite(v)) return false;
    const HighsVarType integrality = model.integrality_[c];
    if (integrality == HighsVarType::kSemiContinuous ||
        integrality == HighsVarType::kSemiInteger) {
      if (std::fabs(v) <= tol) continue;
    }
    if (v < model.col_lower_[c] - tol || v > model.col_upper_[c] + tol)
      return false;
    if (integrality == HighsVarType::kInteger ||
        integrality == HighsVarType::kSemiInteger ||
        integrality == HighsVarType::kImplicitInteger) {
      if (std::fabs(v - std::round(v)) > tol) return false;
    }
  }
  std::vector<double> activity(model.num_row_, 0.0);
  for (HighsInt c = 0; c != model.num_col_; ++c) {
    for (HighsInt el = model.a_matrix_.start_[c];
         el != model.a_matrix_.start_[c + 1]; ++el)
      activity[model.a_matrix_.index_[el]] +=
          model.a_matrix_.value_[el] * sol[c];
  }
  for (HighsInt r = 0; r != model.num_row_; ++r) {
    if (activity[r] < model.row_lower_[r] - tol ||
        activity[r] > model.row_upper_[r] + tol)
      return false;
  }
  return true;
}

// RAII accumulator: every runBenders exit (all fallback returns and the
// normal end) adds its elapsed time to the solve-wide Benders budget,
// so post-restart loops share one total instead of each taking a fresh
// allowance.
struct BendersTimeGuard {
  HighsMipSolverData& data;
  double start;
  explicit BendersTimeGuard(HighsMipSolverData& d)
      : data(d), start(d.mipsolver.timer_.read()) {}
  ~BendersTimeGuard() {
    data.bendersTotalTime += data.mipsolver.timer_.read() - start;
  }
};

bool HighsMipSolverData::runBenders() {
  bendersCoupling.clear();
  // Initial presolve only: post-restart models are LP relaxations plus
  // cuts, where arrowhead structure is a cut artifact. Re-running the
  // fixing loop there drives fix-restart churn (each fix collapses
  // integer activity, triggering another restart and another loop)
  // while the initial pass plus its rescued incumbent carry the value.
  // (The stored coupling hint is intentionally kept across restarts:
  // column indices are stable and it is soft branching guidance.)
  if (numRestarts > 0) return true;
  HighsLp& model = presolvedModel;
  const HighsInt numCol = model.num_col_;
  const HighsInt numRow = model.num_row_;
  if (numCol == 0 || numRow == 0) return true;
  if (numCol < 100) return true;
  if (!mipsolver.options_mip_->mip_decomposition) return true;
  if (!mipsolver.options_mip_->mip_benders) return true;
  if (model.a_matrix_.format_ != MatrixFormat::kColwise)
    model.a_matrix_.ensureColwise();
  const bool logBend = mipsolver.options_mip_->mip_decomposition_logging;
  const HighsLogOptions& logOptions = mipsolver.options_mip_->log_options;
  if (logBend) {
    std::string mdbg;
    char mbuf[200];
    snprintf(mbuf, sizeof(mbuf), "model %dx%d sense=%d",
             (int)numCol, (int)numRow, (int)model.sense_);
    mdbg += mbuf;
    highsLogUser(logOptions, HighsLogType::kInfo, "[Benders] entry%s\n",
                 mdbg.c_str());
  }

  HighsBendersCandidate cand;
  if (!findBendersSeparator(model, cand)) {
    if (logBend)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Benders] no candidate (%s) -> normal MIP\n",
                   cand.reason.c_str());
    return true;
  }
  BendersTimeGuard timeGuard(*this);
  // Publish the coupling set for structure-aware branching (cleared on
  // convergence-and-fix below, when S is fixed away).
  bendersCoupling.assign(numCol, 0);
  for (HighsInt c : cand.couplingCols) bendersCoupling[c] = 1;
  // Internal minimization: negate costs for maximization parents so all
  // dual reasoning below uses a single convention.
  const double sign =
      (model.sense_ == ObjSense::kMaximize) ? -1.0 : 1.0;

  const HighsInt nY = (HighsInt)cand.couplingCols.size();
  const HighsInt nB = (HighsInt)cand.blockCols.size();
  std::vector<HighsInt> sposOf(numCol, -1);
  for (HighsInt i = 0; i != nY; ++i) sposOf[cand.couplingCols[i]] = i;

  // Fixed-column activity shifted out of every row (exact: lb == ub).
  std::vector<double> rowShift(numRow, 0.0);
  for (HighsInt c = 0; c != numCol; ++c) {
    if (model.col_lower_[c] != model.col_upper_[c]) continue;
    const double fixval = model.col_lower_[c];
    for (HighsInt el = model.a_matrix_.start_[c];
         el != model.a_matrix_.start_[c + 1]; ++el)
      rowShift[model.a_matrix_.index_[el]] +=
          model.a_matrix_.value_[el] * fixval;
  }

  // Row-wise adjacency over unfixed columns (built once) for exact
  // entry classification: every nonzero of a block/master row is either
  // fixed (already shifted), separator-owned, or block-owned. Anything
  // else means the separator is invalid and aborts the attempt.
  std::vector<HighsInt> browStart(numRow + 1, 0);
  std::vector<HighsInt> browCol;
  std::vector<double> browVal;
  {
    std::vector<HighsInt> cnt(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (model.col_lower_[c] == model.col_upper_[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el)
        ++cnt[model.a_matrix_.index_[el]];
    }
    for (HighsInt r = 0; r != numRow; ++r)
      browStart[r + 1] = browStart[r] + cnt[r];
    browCol.assign(browStart[numRow], -1);
    browVal.assign(browStart[numRow], 0.0);
    std::vector<HighsInt> fill(numRow, 0);
    for (HighsInt c = 0; c != numCol; ++c) {
      if (model.col_lower_[c] == model.col_upper_[c]) continue;
      for (HighsInt el = model.a_matrix_.start_[c];
           el != model.a_matrix_.start_[c + 1]; ++el) {
        HighsInt r = model.a_matrix_.index_[el];
        HighsInt p = browStart[r] + fill[r]++;
        browCol[p] = c;
        browVal[p] = model.a_matrix_.value_[el];
      }
    }
  }

  struct BendBlock {
    std::vector<HighsInt> cols;
    std::vector<HighsInt> rows;
    // Per block-row: (separator position, value) coupling entries.
    std::vector<std::vector<std::pair<HighsInt, double>>> coupling;
    // Per block-column: (block-row position, value) entries, reused in
    // every iteration when the shifted subproblem is assembled.
    std::vector<std::vector<std::pair<HighsInt, double>>> colEntries;
    std::vector<double> baseLo;
    std::vector<double> baseHi;
    std::vector<double> cost;  // internal-min costs
    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<HighsVarType> integ;  // parent integrality (for sub-MIPs)
  };
  std::vector<BendBlock> blocks(nB);
  std::vector<HighsInt> colToBlockPos(numCol, -1);
  bool separatorBroken = false;
  for (HighsInt k = 0; k != nB && !separatorBroken; ++k) {
    BendBlock& blk = blocks[k];
    blk.cols = cand.blockCols[k];
    blk.rows = cand.blockRows[k];
    blk.coupling.assign(blk.rows.size(), {});
    blk.colEntries.assign(blk.cols.size(), {});
    blk.baseLo.resize(blk.rows.size());
    blk.baseHi.resize(blk.rows.size());
    blk.cost.resize(blk.cols.size());
    blk.lb.resize(blk.cols.size());
    blk.ub.resize(blk.cols.size());
    blk.integ.resize(blk.cols.size());
    for (size_t j = 0; j != blk.cols.size(); ++j) {
      HighsInt c = blk.cols[j];
      colToBlockPos[c] = (HighsInt)j;
      blk.cost[j] = sign * model.col_cost_[c];
      blk.lb[j] = model.col_lower_[c];
      blk.ub[j] = model.col_upper_[c];
      blk.integ[j] = model.integrality_[c];
    }
    for (size_t i = 0; i != blk.rows.size() && !separatorBroken; ++i) {
      HighsInt r = blk.rows[i];
      blk.baseLo[i] = model.row_lower_[r] == -kHighsInf
                          ? -kHighsInf
                          : model.row_lower_[r] - rowShift[r];
      blk.baseHi[i] = model.row_upper_[r] == kHighsInf
                          ? kHighsInf
                          : model.row_upper_[r] - rowShift[r];
      for (HighsInt e = browStart[r]; e != browStart[r + 1]; ++e) {
        HighsInt c = browCol[e];
        if (sposOf[c] >= 0) {
          blk.coupling[i].emplace_back(sposOf[c], browVal[e]);
        } else if (colToBlockPos[c] >= 0) {
          blk.colEntries[colToBlockPos[c]].emplace_back((HighsInt)i,
                                                        browVal[e]);
        } else {
          separatorBroken = true;
          break;
        }
      }
    }
    for (HighsInt c : blk.cols) colToBlockPos[c] = -1;
  }
  if (separatorBroken) {
    if (logBend)
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Benders] block row touches foreign column -> normal MIP\n");
    return true;
  }

  // Master base: separator columns (internal costs, original
  // bounds/integrality) plus theta per block with a trivial valid lower
  // bound (block LP without rows can only underestimate block cost).
  std::vector<double> sepCost(nY);
  for (HighsInt i = 0; i != nY; ++i)
    sepCost[i] = sign * model.col_cost_[cand.couplingCols[i]];
  std::vector<double> thetaLb(nB, 0.0);
  for (HighsInt k = 0; k != nB; ++k) {
    double lb = 0.0;
    bool bounded = true;
    for (size_t j = 0; j != blocks[k].cols.size(); ++j) {
      const double cj = blocks[k].cost[j];
      if (cj > 0) {
        if (!std::isfinite(blocks[k].lb[j])) {
          bounded = false;
          break;
        }
        lb += cj * blocks[k].lb[j];
      } else if (cj < 0) {
        if (!std::isfinite(blocks[k].ub[j])) {
          bounded = false;
          break;
        }
        lb += cj * blocks[k].ub[j];
      }
    }
    thetaLb[k] = bounded ? lb : -kHighsInf;
  }
  // Master rows: rows fully inside separator/fixed (shifted, S entries).
  struct MasterRow {
    std::vector<HighsInt> spos;
    std::vector<double> val;
    double lo;
    double hi;
  };
  std::vector<MasterRow> masterBaseRows;
    for (HighsInt r : cand.masterRows) {
    MasterRow mr;
    mr.lo = model.row_lower_[r] == -kHighsInf
                ? -kHighsInf
                : model.row_lower_[r] - rowShift[r];
    mr.hi = model.row_upper_[r] == kHighsInf
                ? kHighsInf
                : model.row_upper_[r] - rowShift[r];
    bool touchesBlock = false;
    for (HighsInt e = browStart[r]; e != browStart[r + 1]; ++e) {
      HighsInt c = browCol[e];
      if (sposOf[c] >= 0) {
        mr.spos.push_back(sposOf[c]);
        mr.val.push_back(browVal[e]);
      } else {
        // A block column in a master row means the separator is
        // invalid: abort (defensive, should not happen).
        touchesBlock = true;
        break;
      }
    }
    if (touchesBlock) {
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] master row touches block column -> "
                     "normal MIP\n");
      return true;
    }
    masterBaseRows.push_back(std::move(mr));
  }

  if (logBend) {
    HighsInt numLpBlocks = 0;
    HighsInt blockCols = 0;
    for (size_t k = 0; k != cand.blockCols.size(); ++k) {
      numLpBlocks += (cand.blockIsLp[k] != 0);
      blockCols += (HighsInt)cand.blockCols[k].size();
    }
    // Conservative benefit/cost heuristic (Part 12): benefit scales with
    // the columns Benders decides (coupling fixed plus blocks decoupled
    // into independent subproblems); cost scales with solves per
    // iteration times the iteration cap. Caps are the enforcement.
    highsLogUser(logOptions, HighsLogType::kInfo,
                 "[Benders] candidate: %d coupling cols, %d blocks (%d LP, "
                 "%d MIP, %d block cols), %d master rows; score: benefit~%d "
                 "cols decided, cost~%d solves/iter x %d iters\n",
                 (int)nY, (int)nB, (int)numLpBlocks, (int)(nB - numLpBlocks),
                 (int)blockCols, (int)masterBaseRows.size(),
                 (int)(nY + blockCols), (int)(nB + 1),
                 (int)mipsolver.options_mip_->mip_benders_max_iterations);
  }

  struct BendCut {
    HighsInt block;  // -1 = feasibility cut (no theta)
    bool le = false;  // true: a*y <= rhs; false: [theta +] a*y >= rhs
    bool aux = false;  // true: from the auxiliary LP, not a Farkas ray
    bool lshaped = false;  // true: integer L-shaped cut (binary coupling)
    // Cut-management priority (SCIP order: aux > feas > opt > nogood >
    // integer) and post-scale violation, used only when the total cut
    // cap (mip_benders_max_cuts) trims an iteration's new cuts.
    int prio = 0;
    double viol = 0.0;
    std::vector<double> ay;
    double rhs;
  };
  std::vector<BendCut> cuts;
  const HighsInt maxIter = std::max<HighsInt>(
      1, mipsolver.options_mip_->mip_benders_max_iterations);
  const double feastol = mipsolver.options_mip_->mip_feasibility_tolerance;
  // Rescue channel: publish a verified presolved-space composition as a
  // native MIP-start incumbent (mirrors the Lagrangian injection tail:
  // postsolve on the production path, re-check against the original
  // model, publish only if feasible there; checkAddSolution picks it up
  // in runSetup; infeasible candidates are silently dropped).
  auto injectBendersIncumbent = [&](const std::vector<double>& sol) -> void {
    if (!mipsolver.options_mip_->mip_benders_incumbent) return;
    HighsSolution injsol;
    injsol.col_value = sol;
    injsol.value_valid = true;
    injsol.dual_valid = false;
    HighsBasis injbasis;
    injbasis.valid = false;
    // NOTE: thread_safe=false is the production path (used at every
    // solve end). Calling it mid-presolve is safe: it resets its cursor
    // and only reads the stacks, so the final postsolve re-walks
    // identically. Full ctest (exact solution checks) guards this claim.
    postSolveStack.undo(*mipsolver.options_mip_, injsol, injbasis, -1,
                        false);
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
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] injected incumbent (obj %.6g)\n",
                     double(injObj));
    } else if (logBend) {
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Benders] postsolved incumbent infeasible (%.2g, %.2g, "
                   "%.2g) -> dropped\n",
                   boundViol, rowViol, intViol);
    }
  };
  // Exact power-of-two cut normalization to unit max-|ay| (shared
  // hygiene for every cut family): scaling (ay, rhs) by 2^k is bit-exact
  // short of overflow, so already-passed violation gates are unaffected;
  // only master-LP numerics improve. Returns the scale, or 0.0 when the
  // cut must be dropped (non-finite coefficients or scaling overflow).
  auto normalizeCut = [&](BendCut& cut) -> double {
    // Opt-in hygiene (default off = pre-hygiene dynamics bit-exact).
    if (!mipsolver.options_mip_->mip_benders_cut_hygiene) return 1.0;
    double mx = 0.0;
    for (HighsInt i = 0; i != nY; ++i)
      mx = std::max(mx, std::fabs(cut.ay[i]));
    if (mx == 0.0) return 1.0;  // theta-only bound: keep unscaled
    if (!std::isfinite(mx)) return 0.0;
    int e = 0;
    std::frexp(mx, &e);
    const double s = std::ldexp(1.0, -e);
    if (!std::isfinite(s) || s == 0.0) return 0.0;
    for (HighsInt i = 0; i != nY; ++i) {
      cut.ay[i] *= s;
      if (!std::isfinite(cut.ay[i])) return 0.0;
    }
    cut.rhs *= s;
    if (!std::isfinite(cut.rhs)) return 0.0;
    return s;
  };
  // Exact-duplicate suppression against kept cuts (same shape and
  // bitwise coefficients, e.g. a revisited y*): duplicates add nothing
  // mathematically, only master-LP degeneracy.
  auto isDuplicateCut = [&](const BendCut& cut) -> bool {
    for (const BendCut& e : cuts) {
      if (e.block != cut.block || e.le != cut.le || e.aux != cut.aux ||
          e.lshaped != cut.lshaped || e.prio != cut.prio ||
          e.rhs != cut.rhs)
        continue;
      bool same = true;
      for (HighsInt i = 0; i != nY; ++i) {
        if (e.ay[i] != cut.ay[i]) {
          same = false;
          break;
        }
      }
      if (same) return true;
    }
    return false;
  };

  double UB = kHighsInf;  // internal space
  std::vector<double> bestY;
  std::vector<std::vector<double>> bestBlockSol(nB);
  bool hasBest = false;
  HighsInt numIter = 0;
  bool converged = false;
  // Stall discipline (SCIP stalllimit-style): the master lower bound is
  // monotone (cuts only accumulate), so iterations without LB progress
  // learn nothing new and only burn subproblem solves. This only stops
  // the loop EARLIER than max_iterations; cut/fixing logic is unchanged.
  double bestLB = -kHighsInf;
  HighsInt lbStall = 0;
  const HighsInt stallLimit = std::max<HighsInt>(
      1, mipsolver.options_mip_->mip_benders_stall_limit);
  // Total time budget (Lagrangian max_time-style), shared across all
  // runBenders calls of one solve: Benders must never starve the parent
  // search (100 iters x sub-MIPs per loop, times post-restart loops, is
  // otherwise unbounded and can drive fix-restart churn). On exhaustion
  // the loop falls back with whatever UB the rescue can inject; the
  // parent keeps the remaining budget.
  const double bendStart = mipsolver.timer_.read();
  const double bendMaxTime = mipsolver.options_mip_->mip_benders_max_time;

  std::vector<double> y(nY, 0.0);
  const HighsInt nMasterCol = nY + nB;
  for (HighsInt iter = 0; iter != maxIter; ++iter) {
    const double bendSpent =
        bendersTotalTime + mipsolver.timer_.read() - bendStart;
    if (bendSpent >= bendMaxTime) {
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] time budget %.1fs exhausted (%.1fs used) -> "
                     "normal MIP\n",
                     bendMaxTime, bendSpent);
      break;
    }
    if (mipsolver.options_mip_->time_limit < kHighsInf &&
        mipsolver.timer_.read() >= mipsolver.options_mip_->time_limit)
      break;
    // ---- master problem ----
    HighsLp master;
    master.num_col_ = nMasterCol;
    master.num_row_ = (HighsInt)(masterBaseRows.size() + cuts.size());
    master.sense_ = ObjSense::kMinimize;
    master.offset_ = 0.0;
    master.a_matrix_.format_ = MatrixFormat::kColwise;
    master.a_matrix_.start_.assign(nMasterCol + 1, 0);
    master.col_cost_.assign(nMasterCol, 0.0);
    master.col_lower_.assign(nMasterCol, -kHighsInf);
    master.col_upper_.assign(nMasterCol, kHighsInf);
    master.integrality_.assign(nMasterCol, HighsVarType::kContinuous);
    for (HighsInt i = 0; i != nY; ++i) {
      HighsInt c = cand.couplingCols[i];
      master.col_cost_[i] = sepCost[i];
      master.col_lower_[i] = model.col_lower_[c];
      master.col_upper_[i] = model.col_upper_[c];
      master.integrality_[i] = model.integrality_[c];
    }
    for (HighsInt k = 0; k != nB; ++k) {
      master.col_cost_[nY + k] = 1.0;
      master.col_lower_[nY + k] = thetaLb[k];
    }
    // Row-wise assembly, then transpose to colwise via per-col lists.
    std::vector<std::vector<std::pair<HighsInt, double>>> colEntries(
        nMasterCol);
    std::vector<double> mLo;
    std::vector<double> mHi;
    mLo.reserve(master.num_row_);
    mHi.reserve(master.num_row_);
    for (const MasterRow& mr : masterBaseRows) {
      for (size_t t = 0; t != mr.spos.size(); ++t)
        colEntries[mr.spos[t]].emplace_back((HighsInt)mLo.size(),
                                            mr.val[t]);
      mLo.push_back(mr.lo);
      mHi.push_back(mr.hi);
    }
    for (const BendCut& cut : cuts) {
      for (HighsInt i = 0; i != nY; ++i) {
        if (cut.ay[i] != 0.0)
          colEntries[i].emplace_back((HighsInt)mLo.size(), cut.ay[i]);
      }
      if (cut.block >= 0)
        colEntries[nY + cut.block].emplace_back((HighsInt)mLo.size(), 1.0);
      if (cut.le) {
        mLo.push_back(-kHighsInf);
        mHi.push_back(cut.rhs);
      } else {
        mLo.push_back(cut.rhs);
        mHi.push_back(kHighsInf);
      }
    }
    master.row_lower_ = std::move(mLo);
    master.row_upper_ = std::move(mHi);
    for (HighsInt j = 0; j != nMasterCol; ++j) {
      // Deterministic row order within each column.
      std::sort(colEntries[j].begin(), colEntries[j].end(),
                [](const std::pair<HighsInt, double>& a,
                   const std::pair<HighsInt, double>& b) {
                  return a.first < b.first;
                });
      for (const auto& e : colEntries[j]) {
        master.a_matrix_.index_.push_back(e.first);
        master.a_matrix_.value_.push_back(e.second);
      }
      master.a_matrix_.start_[j + 1] =
          (HighsInt)master.a_matrix_.index_.size();
    }

    HighsOptions suboptions = *mipsolver.options_mip_;
    suboptions.output_flag = false;
    suboptions.threads = 1;
    suboptions.mip_max_nodes = 20000;
    suboptions.mip_max_leaves = 2000;
    suboptions.mip_detect_symmetry = false;
    suboptions.random_seed = 0;
    suboptions.mip_rel_gap = 0.0;
    suboptions.mip_abs_gap = 0.0;
    suboptions.mip_feasibility_tolerance = 1e-9;
    {
      double remaining =
          mipsolver.options_mip_->time_limit - mipsolver.timer_.read();
      suboptions.time_limit = std::min(10.0, remaining);
    }
    HighsSolution solution;
    solution.value_valid = false;
    solution.dual_valid = false;
    HighsMipSolver masterSolver(*mipsolver.callback_, suboptions, master,
                                solution, true, mipsolver.submip_level + 1);
    masterSolver.setProfiling(mipsolver.profiling_);
    masterSolver.initialiseTerminator(mipsolver);
    masterSolver.run();
    if (masterSolver.modelstatus_ != HighsModelStatus::kOptimal) {
      if (masterSolver.modelstatus_ == HighsModelStatus::kInfeasible &&
          cuts.empty()) {
        // No cuts yet: master rows are original rows, so an infeasible
        // master proves the whole model infeasible.
        mipsolver.modelstatus_ = HighsModelStatus::kInfeasible;
        if (logBend)
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[Benders] master infeasible without cuts -> "
                       "globally infeasible\n");
        return true;
      }
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] master status %d, aborting -> normal MIP\n",
                     (int)masterSolver.modelstatus_);
      return true;
    }
    const std::vector<double>& masterSol = masterSolver.solution_;
    if ((HighsInt)masterSol.size() != nMasterCol) {
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] master solution size %d != %d -> normal MIP\n",
                     (int)masterSol.size(), (int)nMasterCol);
      return true;
    }
    for (HighsInt i = 0; i != nY; ++i) {
      if (!std::isfinite(masterSol[i])) {
        if (logBend)
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[Benders] master solution non-finite -> normal MIP\n");
        return true;
      }
      y[i] = masterSol[i];
    }
    const double LB = masterSolver.solution_objective_;
    std::vector<double> thetaStar(nB);
    for (HighsInt k = 0; k != nB; ++k) thetaStar[k] = masterSol[nY + k];
    ++numIter;
    if (logBend) {
      std::string ydbg;
      char ybuf[64];
      for (HighsInt i = 0; i != std::min<HighsInt>(nY, 12); ++i) {
        snprintf(ybuf, sizeof(ybuf), " y%d=%.4g", (int)i, y[i]);
        ydbg += ybuf;
      }
      highsLogUser(logOptions, HighsLogType::kInfo, "[Benders] yhat:%s\n",
                   ydbg.c_str());
    }

    // Gap check against the best known feasible composition.
    const double gapTol = 1e-7 * std::max(1.0, std::fabs(UB));
    if (hasBest && UB - LB <= gapTol) {
      converged = true;
      break;
    }
    // Stall check: count consecutive iterations without LB improvement.
    const double lbTol = 1e-9 * std::max(1.0, std::fabs(LB));
    if (LB > bestLB + lbTol) {
      bestLB = LB;
      lbStall = 0;
    } else if (++lbStall >= stallLimit) {
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] lower bound stalled for %d iters -> "
                     "normal MIP\n",
                     (int)lbStall);
      break;
    }
    // ---- subproblems ----
    bool allFeasible = true;
    bool anyCut = false;
    std::vector<BendCut> newCuts;  // this iteration's cuts, priority-merged
    std::vector<std::vector<double>> blockSol(nB);
    std::vector<double> blockObj(nB, kHighsInf);
    const double cutTol = 1e-6;
    // Auxiliary feasibility-cut fallback (SCIP feasalt-style, clean-room):
    // an always-feasible LP minimizing the sum of artificials that relax
    // the shifted block rows. Its minimum q* > 0 proves the block
    // infeasible, and its optimal duals yield a valid >= cut (same shape
    // as an optimality cut but with no theta): ay*y >= rhs with
    // ay = pi*C, rhs = pi*(base bounds) + pi*(bound-row bounds), since
    // weak duality gives 0 >= D(y) for every y with a feasible block.
    // Returns true iff a violated cut was pushed; any failure returns
    // false so the caller falls back exactly as if no ray existed.
    auto pushAuxFeasCut = [&](HighsInt kk, const BendBlock& bblk,
                              const HighsLp& ssublp) -> bool {
      if (!mipsolver.options_mip_->mip_benders_feas_aux) return false;
      const HighsInt nbC = (HighsInt)bblk.cols.size();
      const HighsInt nbR = (HighsInt)bblk.rows.size();
      const HighsInt nSubR = ssublp.num_row_;
      for (HighsInt j = 0; j != nbC; ++j) {
        if (bblk.lb[j] > bblk.ub[j]) return false;
      }
      // One artificial per finite block-row side: -1 on upper-relaxed
      // rows (a*x - v <= hi), +1 on lower-relaxed rows (a*x + v >= lo).
      // Bound rows need none: any x within [lb, ub] satisfies them, so
      // the auxiliary LP below is always feasible (and bounded by 0).
      std::vector<HighsInt> artRow;
      std::vector<double> artCoef;
      for (HighsInt i = 0; i != nbR; ++i) {
        if (ssublp.row_upper_[i] != kHighsInf) {
          artRow.push_back(i);
          artCoef.push_back(-1.0);
        }
        if (ssublp.row_lower_[i] != -kHighsInf) {
          artRow.push_back(i);
          artCoef.push_back(1.0);
        }
      }
      const HighsInt nArt = (HighsInt)artRow.size();
      if (nArt == 0) return false;
      HighsLp auxlp;
      auxlp.num_col_ = nbC + nArt;
      auxlp.num_row_ = nSubR;
      auxlp.sense_ = ObjSense::kMinimize;
      auxlp.offset_ = 0.0;
      auxlp.a_matrix_.format_ = MatrixFormat::kColwise;
      auxlp.a_matrix_.start_.assign(auxlp.num_col_ + 1, 0);
      auxlp.col_cost_.assign(auxlp.num_col_, 0.0);
      auxlp.col_lower_.assign(auxlp.num_col_, -kHighsInf);
      auxlp.col_upper_.assign(auxlp.num_col_, kHighsInf);
      auxlp.integrality_.assign(auxlp.num_col_, HighsVarType::kContinuous);
      auxlp.row_lower_ = ssublp.row_lower_;
      auxlp.row_upper_ = ssublp.row_upper_;
      for (HighsInt j = 0; j != nbC; ++j) {
        for (HighsInt el = ssublp.a_matrix_.start_[j];
             el != ssublp.a_matrix_.start_[j + 1]; ++el) {
          auxlp.a_matrix_.index_.push_back(ssublp.a_matrix_.index_[el]);
          auxlp.a_matrix_.value_.push_back(ssublp.a_matrix_.value_[el]);
        }
        auxlp.a_matrix_.start_[j + 1] =
            (HighsInt)auxlp.a_matrix_.index_.size();
      }
      for (HighsInt a = 0; a != nArt; ++a) {
        auxlp.col_cost_[nbC + a] = 1.0;
        auxlp.col_lower_[nbC + a] = 0.0;
        auxlp.col_upper_[nbC + a] = kHighsInf;
        auxlp.a_matrix_.index_.push_back(artRow[a]);
        auxlp.a_matrix_.value_.push_back(artCoef[a]);
        auxlp.a_matrix_.start_[nbC + a + 1] =
            (HighsInt)auxlp.a_matrix_.index_.size();
      }
      double remaining =
          mipsolver.options_mip_->time_limit - mipsolver.timer_.read();
      HighsSubLpResult aux =
          solveSubLp(auxlp, std::min(10.0, remaining));
      if (aux.status != HighsModelStatus::kOptimal) return false;
      if (!aux.dualValid || (HighsInt)aux.colSol.size() != auxlp.num_col_ ||
          (HighsInt)aux.rowDual.size() != nSubR)
        return false;
      const double auxTol = 1e-9 * std::max(1.0, std::fabs(aux.obj));
      if (!(aux.obj > auxTol)) return false;
      // Tightness gate on the auxiliary duals (same pairing convention
      // and tolerances as the optimality-cut gate).
      const std::vector<double>& pi = aux.rowDual;
      double dualObj = 0.0;
      bool dualOk = true;
      for (HighsInt i = 0; i != nSubR; ++i) {
        if (pi[i] > 0) {
          if (auxlp.row_lower_[i] == -kHighsInf) {
            dualOk = false;
            break;
          }
          dualObj += pi[i] * auxlp.row_lower_[i];
        } else if (pi[i] < 0) {
          if (auxlp.row_upper_[i] == kHighsInf) {
            dualOk = false;
            break;
          }
          dualObj += pi[i] * auxlp.row_upper_[i];
        }
      }
      const double dualTol = 1e-6 * std::max(1.0, std::fabs(aux.obj));
      if (!dualOk || std::fabs(dualObj - aux.obj) > dualTol) return false;
      BendCut cut;
      cut.block = -1;
      cut.le = false;
      cut.aux = true;
      cut.ay.assign(nY, 0.0);
      for (HighsInt i = 0; i != nbR; ++i) {
        if (pi[i] == 0.0) continue;
        for (const auto& e : bblk.coupling[i])
          cut.ay[e.first] += pi[i] * e.second;
      }
      double rhs = 0.0;
      bool rhsOk = true;
      for (HighsInt i = 0; i != nbR; ++i) {
        if (pi[i] > 0) {
          if (bblk.baseLo[i] == -kHighsInf) {
            rhsOk = false;
            break;
          }
          rhs += pi[i] * bblk.baseLo[i];
        } else if (pi[i] < 0) {
          if (bblk.baseHi[i] == kHighsInf) {
            rhsOk = false;
            break;
          }
          rhs += pi[i] * bblk.baseHi[i];
        }
      }
      if (!rhsOk) return false;
      for (HighsInt i = nbR; i != nSubR; ++i) {
        if (pi[i] > 0)
          rhs += pi[i] * ssublp.row_lower_[i];
        else if (pi[i] < 0)
          rhs += pi[i] * ssublp.row_upper_[i];
      }
      for (HighsInt i = 0; i != nY; ++i)
        if (std::fabs(cut.ay[i]) <= 1e-12) cut.ay[i] = 0.0;
      cut.rhs = rhs;
      double lhs = 0.0;
      for (HighsInt i = 0; i != nY; ++i) lhs += cut.ay[i] * y[i];
      const double violTol = cutTol * std::max(1.0, std::fabs(rhs));
      if (!(rhs - lhs > violTol)) return false;
      const double rawViol = rhs - lhs;
      const double s = normalizeCut(cut);
      if (s == 0.0) return false;
      cut.prio = 4;
      cut.viol = s * rawViol;
      HighsInt nnz = 0;
      for (HighsInt i = 0; i != nY; ++i)
        if (cut.ay[i] != 0.0) ++nnz;
      const double logRhs = cut.rhs;
      const double logViol = cut.viol;
      newCuts.push_back(std::move(cut));
      if (logBend) {
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] block %d aux feasibility cut: nnz=%d rhs=%.6g "
                     "viol=%.4g\n",
                     (int)kk, (int)nnz, logRhs, logViol);
      }
      return true;
    };
    for (HighsInt k = 0; k != nB; ++k) {
      const BendBlock& blk = blocks[k];
      const HighsInt nbC = (HighsInt)blk.cols.size();
      const HighsInt nbR = (HighsInt)blk.rows.size();
      // Coupling shift for this master solution.
      std::vector<double> shift(nbR, 0.0);
      for (HighsInt i = 0; i != nbR; ++i) {
        double s = 0.0;
        for (const auto& e : blk.coupling[i]) s += e.second * y[e.first];
        shift[i] = s;
      }
      // Sub-LP: block rows (shifted) plus explicit finite-bound rows.
      // Columns are free: all dual mass sits on rows, a single source
      // for both cut types.
      HighsLp sublp;
      sublp.num_col_ = nbC;
      sublp.sense_ = ObjSense::kMinimize;
      sublp.offset_ = 0.0;
      sublp.a_matrix_.format_ = MatrixFormat::kColwise;
      sublp.a_matrix_.start_.assign(nbC + 1, 0);
      sublp.col_cost_ = blk.cost;
      sublp.col_lower_.assign(nbC, -kHighsInf);
      sublp.col_upper_.assign(nbC, kHighsInf);
      sublp.integrality_.assign(nbC, HighsVarType::kContinuous);
      std::vector<double> subLo;
      std::vector<double> subHi;
      subLo.reserve(nbR + 2 * nbC);
      subHi.reserve(nbR + 2 * nbC);
      // Block-row entry lookup comes from the precomputed per-column
      // lists (built once during candidate construction).
      std::vector<std::vector<std::pair<HighsInt, double>>> subColEntries =
          blk.colEntries;
      for (HighsInt i = 0; i != nbR; ++i) {
        subLo.push_back(blk.baseLo[i] == -kHighsInf
                            ? -kHighsInf
                            : blk.baseLo[i] - shift[i]);
        subHi.push_back(blk.baseHi[i] == kHighsInf
                            ? kHighsInf
                            : blk.baseHi[i] - shift[i]);
      }
      // Explicit bound rows (finite bounds only).
      for (HighsInt j = 0; j != nbC; ++j) {
        if (std::isfinite(blk.lb[j])) {
          subColEntries[j].emplace_back((HighsInt)subLo.size(), -1.0);
          subLo.push_back(-kHighsInf);
          subHi.push_back(-blk.lb[j]);
        }
        if (std::isfinite(blk.ub[j])) {
          subColEntries[j].emplace_back((HighsInt)subLo.size(), 1.0);
          subLo.push_back(-kHighsInf);
          subHi.push_back(blk.ub[j]);
        }
      }
      sublp.num_row_ = (HighsInt)subLo.size();
      sublp.row_lower_ = std::move(subLo);
      sublp.row_upper_ = std::move(subHi);
      for (HighsInt j = 0; j != nbC; ++j) {
        std::sort(subColEntries[j].begin(), subColEntries[j].end(),
                  [](const std::pair<HighsInt, double>& a,
                     const std::pair<HighsInt, double>& b) {
                    return a.first < b.first;
                  });
        for (const auto& e : subColEntries[j]) {
          sublp.a_matrix_.index_.push_back(e.first);
          sublp.a_matrix_.value_.push_back(e.second);
        }
        sublp.a_matrix_.start_[j + 1] =
            (HighsInt)sublp.a_matrix_.index_.size();
      }
      double remaining =
          mipsolver.options_mip_->time_limit - mipsolver.timer_.read();
      HighsSubLpResult res =
          solveSubLp(sublp, std::min(10.0, remaining));
      if (res.status == HighsModelStatus::kOptimal) {
        if (!res.dualValid ||
            (HighsInt)res.colSol.size() != nbC ||
            (HighsInt)res.rowDual.size() != sublp.num_row_) {
          if (logBend)
            highsLogUser(logOptions, HighsLogType::kInfo,
                         "[Benders] block %d optimal but dual unusable "
                         "(dual_valid=%d) -> normal MIP\n",
                         (int)k, (int)res.dualValid);
          return true;  // cannot build a trusted cut: fallback
        }
        // Tightness gate: the dual objective at this point must
        // reproduce the primal optimum. HiGHS min-LP convention (read off
        // real duals): pi > 0 pairs with the LOWER bound, pi < 0 with the
        // UPPER bound. Any systematic convention error trips here and
        // aborts instead of cutting valid points.
        const std::vector<double>& pi = res.rowDual;
        double dualObj = 0.0;
        bool dualOk = true;
        for (HighsInt i = 0; i != sublp.num_row_; ++i) {
          if (pi[i] > 0) {
            if (sublp.row_lower_[i] == -kHighsInf) {
              dualOk = false;
              break;
            }
            dualObj += pi[i] * sublp.row_lower_[i];
          } else if (pi[i] < 0) {
            if (sublp.row_upper_[i] == kHighsInf) {
              dualOk = false;
              break;
            }
            dualObj += pi[i] * sublp.row_upper_[i];
          }
        }
        const double dualTol = 1e-6 * std::max(1.0, std::fabs(res.obj));
        const double dualLooseTol = 1e-4 * std::max(1.0, std::fabs(res.obj));
        const double dualErr = std::fabs(dualObj - res.obj);
        if (!dualOk || dualErr > dualLooseTol) {
          // Loose failure: systematic convention error, not noise.
          // Abort rather than risk an invalid cut.
          if (logBend)
            highsLogUser(logOptions, HighsLogType::kInfo,
                         "[Benders] block %d dual mismatch "
                         "(dualobj=%.6g primal=%.6g) -> normal MIP\n",
                         (int)k, dualObj, res.obj);
          return true;
        }
        // Tight failure within loose tolerance: real-world dual noise
        // (large unscaled models). Skip only this cut and continue with
        // the block's primal solution for the upper bound; convergence
        // (and hence any fixing) still needs the gap to close.
        const bool cutTrusted = (dualOk && dualErr <= dualTol);
        if (!cutTrusted && logBend)
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[Benders] block %d dual noisy (err=%.3g), skipping "
                       "cut\n",
                       (int)k, dualErr);
        // Optimality cut: theta_k + (pi*C) y >= pi*b(base) + bound terms.
        // Built only from trusted duals (see gate above).
        if (cutTrusted) {
          BendCut cut;
          cut.block = k;
          cut.ay.assign(nY, 0.0);
          for (HighsInt i = 0; i != nbR; ++i) {
            if (pi[i] == 0.0) continue;
            for (const auto& e : blk.coupling[i])
              cut.ay[e.first] += pi[i] * e.second;
          }
          double rhs = 0.0;
          bool rhsOk = true;
          for (HighsInt i = 0; i != nbR; ++i) {
            if (pi[i] > 0) {
              if (blk.baseLo[i] == -kHighsInf) {
                rhsOk = false;
                break;
              }
              rhs += pi[i] * blk.baseLo[i];
            } else if (pi[i] < 0) {
              if (blk.baseHi[i] == kHighsInf) {
                rhsOk = false;
                break;
              }
              rhs += pi[i] * blk.baseHi[i];
            }
          }
          if (!rhsOk) {
            if (logBend)
              highsLogUser(logOptions, HighsLogType::kInfo,
                           "[Benders] block %d cut needs infinite bound -> "
                           "normal MIP\n",
                           (int)k);
            return true;
          }
          for (HighsInt i = nbR; i != sublp.num_row_; ++i) {
            if (pi[i] > 0)
              rhs += pi[i] * sublp.row_lower_[i];
            else if (pi[i] < 0)
              rhs += pi[i] * sublp.row_upper_[i];
          }
          // Coefficient hygiene.
          for (HighsInt i = 0; i != nY; ++i)
            if (std::fabs(cut.ay[i]) <= 1e-12) cut.ay[i] = 0.0;
          cut.rhs = rhs;
          double cutVal = rhs;
          for (HighsInt i = 0; i != nY; ++i) cutVal -= cut.ay[i] * y[i];
          const double violTol = cutTol * std::max(1.0, std::fabs(rhs));
          if (thetaStar[k] < cutVal - violTol) {
            const double rawViol = cutVal - thetaStar[k];
            const double s = normalizeCut(cut);
            if (s > 0.0) {
              cut.prio = 2;
              cut.viol = s * rawViol;
              newCuts.push_back(std::move(cut));
            }
          }
        }
        const bool lpBlock = (bool)cand.blockIsLp[k];
        if (lpBlock) {
          blockSol[k] = res.colSol;
          blockObj[k] = res.obj;
        } else {
          // Integer block: the LP solution above is only a relaxation
          // (its cut stays valid). Solve the true sub-MIP for the upper
          // bound; anything but a proven optimal/infeasible outcome
          // aborts to the normal MIP path.
          HighsLp msublp = sublp;
          msublp.integrality_ = blk.integ;
          HighsOptions msuboptions = *mipsolver.options_mip_;
          msuboptions.output_flag = false;
          msuboptions.threads = 1;
          msuboptions.mip_max_nodes = 20000;
          msuboptions.mip_max_leaves = 2000;
          msuboptions.mip_detect_symmetry = false;
          msuboptions.random_seed = 0;
          msuboptions.mip_rel_gap = 0.0;
          msuboptions.mip_abs_gap = 0.0;
          msuboptions.mip_feasibility_tolerance = 1e-9;
          {
            double remaining =
                mipsolver.options_mip_->time_limit - mipsolver.timer_.read();
            msuboptions.time_limit = std::min(10.0, remaining);
          }
          HighsSolution msolution;
          msolution.value_valid = false;
          msolution.dual_valid = false;
          HighsMipSolver msubsolver(*mipsolver.callback_, msuboptions, msublp,
                                    msolution, true,
                                    mipsolver.submip_level + 1);
          msubsolver.setProfiling(mipsolver.profiling_);
          msubsolver.initialiseTerminator(mipsolver);
          msubsolver.run();
          if (msubsolver.modelstatus_ == HighsModelStatus::kOptimal) {
            if (!verifyComponentSolution(msublp, msubsolver.solution_)) {
              if (logBend)
                highsLogUser(logOptions, HighsLogType::kInfo,
                             "[Benders] block %d sub-MIP solution failed "
                             "verification -> normal MIP\n",
                             (int)k);
              return true;
            }
            blockSol[k] = msubsolver.solution_;
            blockObj[k] = msubsolver.solution_objective_;
            // Integer L-shaped cut (Laporte-Louveaux, clean-room): with
            // binary coupling, the sub-MIP optimum z* at the binary y*
            // tightens theta_k via theta_k >= z* - (z* - L)*dev(y),
            // where dev counts coordinates differing from y* and L is
            // the block's global box lower bound (thetaLb, enforced on
            // theta in every master). At y* the cut is tight
            // (theta >= z*); any other binary y has dev >= 1, giving
            // theta >= L, already enforced hence valid. The master only
            // ever produces binary y (MIP-optimal), so no fractional
            // case arises; any doubt skips the cut. Each cut excludes
            // one y*, giving finite convergence on binary coupling.
            if (mipsolver.options_mip_->mip_benders_lshaped &&
                std::isfinite(thetaLb[k])) {
              bool binaryY = true;
              for (HighsInt i = 0; i != nY; ++i) {
                HighsInt c = cand.couplingCols[i];
                if (model.integrality_[c] != HighsVarType::kInteger ||
                    model.col_lower_[c] != 0.0 ||
                    model.col_upper_[c] != 1.0) {
                  binaryY = false;
                  break;
                }
                const double r = std::round(y[i]);
                if ((r != 0.0 && r != 1.0) || std::fabs(y[i] - r) > 1e-9) {
                  binaryY = false;
                  break;
                }
              }
              if (binaryY) {
                const double zstar = msubsolver.solution_objective_;
                const double spread = zstar - thetaLb[k];
                const double lViolTol =
                    cutTol * std::max(1.0, std::fabs(zstar));
                if (spread > lViolTol && thetaStar[k] < zstar - lViolTol) {
                  BendCut lcut;
                  lcut.block = k;
                  lcut.le = false;
                  lcut.lshaped = true;
                  lcut.ay.assign(nY, 0.0);
                  double numOne = 0.0;
                  for (HighsInt i = 0; i != nY; ++i) {
                    if (std::round(y[i]) == 1.0) {
                      lcut.ay[i] = spread;
                      numOne += 1.0;
                    } else {
                      lcut.ay[i] = -spread;
                    }
                  }
                  lcut.rhs = zstar - spread * numOne;
                  const double rawViol = zstar - thetaStar[k];
                  const double s = normalizeCut(lcut);
                  if (s > 0.0) {
                    lcut.prio = 0;
                    lcut.viol = s * rawViol;
                    newCuts.push_back(std::move(lcut));
                  }
                  if (logBend)
                    highsLogUser(logOptions, HighsLogType::kInfo,
                                 "[Benders] block %d L-shaped cut: z*=%.6g "
                                 "L=%.6g\n",
                                 (int)k, zstar, thetaLb[k]);
                }
              }
            }
          } else if (msubsolver.modelstatus_ == HighsModelStatus::kInfeasible) {
            // LP-relaxation feasible but sub-MIP infeasible: only a
            // binary no-good cut can proceed; anything else aborts.
            allFeasible = false;
            bool allBinary = true;
            for (HighsInt i = 0; i != nY; ++i) {
              HighsInt c = cand.couplingCols[i];
              if (model.integrality_[c] != HighsVarType::kInteger ||
                  model.col_lower_[c] != 0.0 || model.col_upper_[c] != 1.0) {
                allBinary = false;
                break;
              }
            }
            if (!allBinary) {
              if (logBend)
                highsLogUser(logOptions, HighsLogType::kInfo,
                             "[Benders] block %d MIP-infeasible with "
                             "non-binary coupling -> normal MIP\n",
                             (int)k);
              return true;
            }
            // No-good cut over binary coupling: the proven-infeasible y
            // admits no feasible completion, so every global solution
            // differs from it in at least one coordinate.
            BendCut ngcut;
            ngcut.block = -1;
            ngcut.le = false;
            ngcut.ay.assign(nY, 0.0);
            HighsInt numOnes = 0;
            bool binaryY = true;
            for (HighsInt i = 0; i != nY; ++i) {
              const double r = std::round(y[i]);
              if (r == 0.0) {
                ngcut.ay[i] = 1.0;
              } else if (r == 1.0) {
                ngcut.ay[i] = -1.0;
                ++numOnes;
              } else {
                binaryY = false;
                break;
              }
            }
            if (!binaryY) return true;
            ngcut.rhs = 1.0 - (double)numOnes;
            double nglhs = 0.0;
            for (HighsInt i = 0; i != nY; ++i) nglhs += ngcut.ay[i] * y[i];
            if (!(nglhs < ngcut.rhs - 1e-9)) {
              if (logBend)
                highsLogUser(logOptions, HighsLogType::kInfo,
                             "[Benders] block %d no-good not violated -> "
                             "normal MIP\n",
                             (int)k);
              return true;
            }
            {
              const double rawViol = ngcut.rhs - nglhs;
              const double s = normalizeCut(ngcut);
              if (s > 0.0) {
                ngcut.prio = 1;
                ngcut.viol = s * rawViol;
                newCuts.push_back(std::move(ngcut));
              }
            }
          } else {
            if (logBend)
              highsLogUser(logOptions, HighsLogType::kInfo,
                           "[Benders] block %d sub-MIP status %d -> "
                           "normal MIP\n",
                           (int)k, (int)msubsolver.modelstatus_);
            return true;
          }
        }
      } else if (res.status == HighsModelStatus::kInfeasible) {
        allFeasible = false;
        // Farkas ray -> feasibility cut (same row-only dual source).
        Highs lpsolver;
        lpsolver.setOptionValue("output_flag", false);
        lpsolver.setOptionValue("presolve", kHighsOffString);
        lpsolver.setOptionValue("solver", kSimplexString);
        bool rayUsable = true;
        if (lpsolver.passModel(sublp) != HighsStatus::kOk) rayUsable = false;
        if (rayUsable && lpsolver.run() != HighsStatus::kOk) rayUsable = false;
        if (rayUsable &&
            lpsolver.getModelStatus() != HighsModelStatus::kInfeasible)
          rayUsable = false;
        std::vector<double> ray(sublp.num_row_, 0.0);
        bool hasRay = false;
        if (rayUsable &&
            (lpsolver.getDualRay(hasRay, ray.data()) != HighsStatus::kOk ||
             !hasRay))
          rayUsable = false;
        if (!rayUsable) {
          // No ray: try the auxiliary fallback before giving up (a pushed
          // aux cut continues with the next block via continue).
          if (pushAuxFeasCut(k, blk, sublp)) continue;
          if (logBend)
            highsLogUser(logOptions, HighsLogType::kInfo,
                         "[Benders] block %d infeasible but no dual ray -> "
                         "normal MIP\n",
                         (int)k);
          return true;
        }
        // Farkas certificate for the shifted rows: with w = -ray and the
        // conflict-proof pairing (upper bound active for w > 0), the
        // infeasibility proof is viol < 0, and the valid cut is the same
        // combination with y-dependent bounds: a*y <= rhs.
        double viol = 0.0;
        for (HighsInt i = 0; i != sublp.num_row_; ++i) {
          const double w = -ray[i];
          if (w > 0)
            viol += w * sublp.row_upper_[i];
          else if (w < 0)
            viol += w * sublp.row_lower_[i];
        }
        if (!(viol < -1e-7 * std::max(1.0, std::fabs(viol)))) {
          // Ray does not prove infeasibility at this point: try the
          // auxiliary fallback before giving up.
          if (pushAuxFeasCut(k, blk, sublp)) continue;
          if (logBend)
            highsLogUser(logOptions, HighsLogType::kInfo,
                         "[Benders] block %d ray violation %g not negative "
                         "-> normal MIP\n",
                         (int)k, viol);
          return true;
        }
        BendCut cut;
        cut.block = -1;
        cut.le = true;
        cut.ay.assign(nY, 0.0);
        double rhs = 0.0;
        bool feasRhsOk = true;
        for (HighsInt i = 0; i != nbR; ++i) {
          const double w = -ray[i];
          if (w == 0.0) continue;
          for (const auto& e : blk.coupling[i]) cut.ay[e.first] += w * e.second;
          if (w > 0) {
            if (blk.baseHi[i] == kHighsInf) {
              feasRhsOk = false;
              break;
            }
            rhs += w * blk.baseHi[i];
          } else {
            if (blk.baseLo[i] == -kHighsInf) {
              feasRhsOk = false;
              break;
            }
            rhs += w * blk.baseLo[i];
          }
        }
        if (!feasRhsOk) {
          if (pushAuxFeasCut(k, blk, sublp)) continue;
          if (logBend)
            highsLogUser(logOptions, HighsLogType::kInfo,
                         "[Benders] block %d feasibility cut needs infinite "
                         "bound -> normal MIP\n",
                         (int)k);
          return true;
        }
        for (HighsInt i = nbR; i != sublp.num_row_; ++i) {
          const double w = -ray[i];
          if (w > 0)
            rhs += w * sublp.row_upper_[i];
          else if (w < 0)
            rhs += w * sublp.row_lower_[i];
        }
        for (HighsInt i = 0; i != nY; ++i)
          if (std::fabs(cut.ay[i]) <= 1e-12) cut.ay[i] = 0.0;
        cut.rhs = rhs;
        // The cut a*y <= rhs must be violated at y (by -viol > 0);
        // anything else signals a broken certificate: auxiliary fallback.
        double lhs = 0.0;
        for (HighsInt i = 0; i != nY; ++i) lhs += cut.ay[i] * y[i];
        const double rayTol = 1e-6 * std::max(1.0, std::fabs(rhs));
        if (lhs > rhs + rayTol) {
          const double rawViol = lhs - rhs;
          const double s = normalizeCut(cut);
          if (s > 0.0) {
            cut.prio = 3;
            cut.viol = s * rawViol;
            HighsInt nnz = 0;
            std::string aydbg;
            char abuf[64];
            for (HighsInt i = 0; i != nY; ++i) {
              if (cut.ay[i] != 0.0) {
                ++nnz;
                snprintf(abuf, sizeof(abuf), " a%d=%.4g", (int)i,
                         cut.ay[i]);
                aydbg += abuf;
              }
            }
            const double logRhs = cut.rhs;
            const double logViol = cut.viol;
            newCuts.push_back(std::move(cut));
            if (logBend) {
              highsLogUser(logOptions, HighsLogType::kInfo,
                           "[Benders] block %d feasibility cut: nnz=%d "
                           "rhs=%.6g viol=%.4g%s\n",
                           (int)k, (int)nnz, logRhs, logViol,
                           aydbg.c_str());
            }
          }
        } else {
          if (pushAuxFeasCut(k, blk, sublp)) continue;
          if (logBend)
            highsLogUser(logOptions, HighsLogType::kInfo,
                         "[Benders] block %d feasibility cut not violated "
                         "at y -> normal MIP\n",
                         (int)k);
          return true;
        }
      } else if (res.status == HighsModelStatus::kUnbounded) {
        // An unbounded block subproblem means the global problem is
        // unbounded along this ray; leave that verdict to normal MIP.
        if (logBend)
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[Benders] block %d unbounded -> normal MIP\n", (int)k);
        return true;
      } else {
        if (logBend)
          highsLogUser(logOptions, HighsLogType::kInfo,
                       "[Benders] block %d status %d -> normal MIP\n", (int)k,
                       (int)res.status);
        return true;  // time limit, error, ...: fallback
      }
    }
    // Priority merge of this iteration's cuts (SCIP cut-management
    // order: aux > feas > opt > nogood > lshaped). Unlimited cap
    // (default): append in generation order, with exact-duplicate
    // suppression only when cut hygiene is on (default off, preserving
    // pre-hygiene loop dynamics bit-exactly). Capped: keep the
    // most-violated cuts first. Dropping valid cuts only slows
    // convergence; the stall limit and fallback cover non-convergence.
    {
      const HighsInt maxCuts =
          mipsolver.options_mip_->mip_benders_max_cuts;
      std::vector<size_t> order(newCuts.size());
      for (size_t i = 0; i != order.size(); ++i) order[i] = i;
      if (maxCuts < kHighsIInf) {
        std::stable_sort(
            order.begin(), order.end(),
            [&](size_t a, size_t b) {
              if (newCuts[a].prio != newCuts[b].prio)
                return newCuts[a].prio > newCuts[b].prio;
              return newCuts[a].viol > newCuts[b].viol;
            });
      }
      HighsInt dropped = 0;
      const HighsInt before = (HighsInt)cuts.size();
      const bool hygiene =
          mipsolver.options_mip_->mip_benders_cut_hygiene;
      for (size_t idx : order) {
        BendCut& nc = newCuts[idx];
        if (hygiene && isDuplicateCut(nc)) {
          ++dropped;
          continue;
        }
        if ((HighsInt)cuts.size() >= maxCuts) {
          ++dropped;
          continue;
        }
        cuts.push_back(std::move(nc));
      }
      if ((HighsInt)cuts.size() > before) anyCut = true;
      if (logBend && dropped > 0)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] iter %d: dropped %d duplicate/excess cuts\n",
                     (int)iter, (int)dropped);
    }
    if (allFeasible) {
      double composed = 0.0;
      for (HighsInt i = 0; i != nY; ++i) composed += sepCost[i] * y[i];
      for (HighsInt k = 0; k != nB; ++k) composed += blockObj[k];
      if (composed < UB) {
        UB = composed;
        bestY = y;
        bestBlockSol = blockSol;
        hasBest = true;
      }
    }
    if (logBend) {
      HighsInt nFeas = 0, nOpt = 0, nNoGood = 0, nAux = 0, nLshaped = 0;
      for (const BendCut& cut : cuts) {
        if (cut.aux)
          ++nAux;
        else if (cut.lshaped)
          ++nLshaped;
        else if (cut.block >= 0)
          ++nOpt;
        else if (cut.le)
          ++nFeas;
        else
          ++nNoGood;
      }
      highsLogUser(logOptions, HighsLogType::kInfo,
                   "[Benders] iter %d: master=%.6g LB=%.6g UB=%.6g cuts=%d "
                   "(feas=%d opt=%d nogood=%d aux=%d lshaped=%d)\n",
                   (int)iter, LB, LB, UB, (int)cuts.size(), (int)nFeas,
                   (int)nOpt, (int)nNoGood, (int)nAux, (int)nLshaped);
    }
    // Post-subsolve gap check with a FRESH tolerance from the current
    // UB (gapTol above predates this iteration's UB update and may still
    // be infinite from the first iteration: using it here would declare
    // convergence on any first feasible UB regardless of the gap).
    if (hasBest) {
      const double postGapTol = 1e-7 * std::max(1.0, std::fabs(UB));
      if (UB - LB <= postGapTol) {
        converged = true;
        break;
      }
    }
    if (!anyCut) break;  // nothing learned: gap decides fix vs fallback
  }

  // Compose the full presolved-space solution whenever the loop produced
  // a feasible composition. A verified composition is injected as a
  // native MIP-start incumbent even on fallback (rescue: the loop's UB
  // would otherwise be discarded); on convergence it is additionally
  // fixed as before.
  //
  // Branch-and-check note: node-level Benders cut enforcement is
  // deliberately NOT implemented. In a full-model B&B every LP point
  // restricts to a feasible block-LP witness (it satisfies all model
  // rows within global bounds), so LP-feasibility separation at nodes
  // is provably vacuous; true branch-and-check needs a master-as-model
  // search, which is out of scope. The primal half (this rescue) is
  // what transfers.
  if (hasBest) {
    std::vector<double> fullSol(numCol, 0.0);
    for (HighsInt c = 0; c != numCol; ++c)
      fullSol[c] = model.col_lower_[c];  // fixed cols (lb == ub) included
    for (HighsInt i = 0; i != nY; ++i) {
      HighsInt c = cand.couplingCols[i];
      double v = bestY[i];
      if (model.integrality_[c] == HighsVarType::kInteger) v = std::round(v);
      fullSol[c] = v;
    }
    for (HighsInt k = 0; k != nB; ++k) {
      for (size_t j = 0; j != blocks[k].cols.size(); ++j)
        fullSol[blocks[k].cols[j]] = bestBlockSol[k][j];
    }
    if (verifyBendersSolution(model, fullSol)) {
      if (converged) {
        HighsInt numFixed = 0;
        for (HighsInt i = 0; i != nY; ++i) {
          HighsInt c = cand.couplingCols[i];
          if (model.col_lower_[c] == model.col_upper_[c]) continue;
          double fixval = fullSol[c];
          if (model.integrality_[c] == HighsVarType::kInteger)
            fixval = std::round(fixval);
          fixval = std::min(std::max(fixval, model.col_lower_[c]),
                            model.col_upper_[c]);
          model.col_lower_[c] = model.col_upper_[c] = fixval;
          ++numFixed;
        }
        const double parentUB = sign * UB + model.offset_;
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] fixed %d coupling columns (%d blocks, %d "
                     "iters, verified obj %.6g)\n",
                     (int)numFixed, (int)nB, (int)numIter, parentUB);
        bendersCoupling.clear();  // S fixed away: nothing to prioritize
        return true;
      }
      // Fallback rescue only: the converged path fixes (no trajectory
      // change from an extra incumbent); here the loop's feasible UB
      // would otherwise be discarded, so offer it as MIP start.
      injectBendersIncumbent(fullSol);
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] gap open after %d iters; incumbent offered "
                     "-> normal MIP\n",
                     (int)numIter);
    } else {
      // Fail-closed: an unverified composition fixes nothing and proves
      // nothing; behave exactly as a non-converged loop.
      converged = false;
      if (logBend)
        highsLogUser(logOptions, HighsLogType::kInfo,
                     "[Benders] composed solution failed verification -> "
                     "normal MIP\n");
    }
  }
  if (!converged && logBend)
    highsLogUser(logOptions, HighsLogType::kInfo,
                 "[Benders] not converged (%d iters) -> normal MIP\n",
                 (int)numIter);
  return true;
}

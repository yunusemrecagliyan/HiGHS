/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_MIP_SOLVER_DATA_H_
#define HIGHS_MIP_SOLVER_DATA_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "lp_data/HConst.h"

#include "mip/HighsCliqueTable.h"
#include "mip/HighsConflictPool.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsDebugSol.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipWorker.h"
#include "mip/HighsNodeQueue.h"
#include "mip/HighsObjectiveFunction.h"
#include "mip/HighsPrimalHeuristics.h"
#include "mip/HighsPseudocost.h"
#include "mip/HighsRedcostFixing.h"
#include "mip/HighsSearch.h"
#include "mip/HighsSeparation.h"
#include "parallel/HighsParallel.h"
#include "presolve/HighsPostsolveStack.h"
#include "presolve/HighsSymmetry.h"
#include "util/HighsTimer.h"

struct HighsPrimaDualIntegral {
  double value;
  double prev_lb;
  double prev_ub;
  double prev_gap;
  double prev_time;
  void initialise();
};

enum MipSolutionSource : int {
  kSolutionSourceNone = -1,
  kSolutionSourceMin = kSolutionSourceNone,
  //  kSolutionSourceInitial, // 0
  kSolutionSourceBranching,           // B
  kSolutionSourceCentralRounding,     // C
  kSolutionSourceFeasibilityPump,     // F
  kSolutionSourceHeuristic,           // H
  kSolutionSourceShifting,            // I
  kSolutionSourceFeasibilityJump,     // J
  kSolutionSourceSubMip,              // L
  kSolutionSourceEmptyMip,            // P
  kSolutionSourceRandomizedRounding,  // R
  kSolutionSourceSolveLp,             // S
  kSolutionSourceEvaluateNode,        // T
  kSolutionSourceUnbounded,           // U
  kSolutionSourceUserSolution,        // X
  kSolutionSourceHighsSolution,       // Y
  kSolutionSourceZiRound,             // Z
  kSolutionSourceTrivialL,            // l
  kSolutionSourceTrivialP,            // p
  kSolutionSourceTrivialU,            // u
  kSolutionSourceTrivialZ,            // z
  kSolutionSourceCleanup,
  kSolutionSourceCount
};

struct HighsMipSolverData {
  HighsMipSolver& mipsolver;

  std::deque<HighsLpRelaxation> lps;
  std::deque<HighsCutPool> cutpools;
  std::deque<HighsConflictPool> conflictpools;
  std::deque<HighsDomain> domains;
  std::deque<HighsPseudocost> pseudocosts;
  std::deque<HighsMipWorker> workers;
  bool parallel_lock;
  std::atomic<int64_t> worker_lp_iterations_stop;

  HighsPrimalHeuristics heuristics;
  HighsCliqueTable cliquetable;
  HighsImplications implications;
  HighsRedcostFixing redcostfixing;
  HighsObjectiveFunction objectiveFunction;
  presolve::HighsPostsolveStack postSolveStack;
  HighsPresolveStatus presolve_status;
  HighsLp presolvedModel;
  bool cliquesExtracted;
  bool rowMatrixSet;
  bool analyticCenterComputed;
  HighsModelStatus analyticCenterStatus;
  bool detectSymmetries;
  HighsInt numRestarts;
  HighsInt numRestartsRoot;
  // Inactive-integer percentage recorded at the last root restart; a new
  // restart is only performed if it strictly reduces this percentage
  // (restarts bring no progress otherwise)
  // Analysis of the first root LP solution: variables that were fractional in
  // it form the "active decision core" used to prioritize strong branching
  // (kept as a record of the analysis framework; not currently referenced)
  double lastRestartInactive = -1.0;
  HighsInt numCliqueEntriesAfterPresolve;
  HighsInt numCliqueEntriesAfterFirstPresolve;

  std::vector<HighsInt> ARstart_;
  std::vector<HighsInt> ARindex_;
  std::vector<double> ARvalue_;
  std::vector<double> maxAbsRowCoef;
  std::vector<uint8_t> rowintegral;
  std::vector<HighsInt> uplocks;
  std::vector<HighsInt> downlocks;
  std::vector<HighsInt> integer_cols;
  std::vector<HighsInt> implint_cols;
  std::vector<HighsInt> integral_cols;
  std::vector<HighsInt> continuous_cols;

  HighsSymmetries symmetries;
  std::shared_ptr<const StabilizerOrbits> globalOrbits;

  double feastol;
  double epsilon;
  double heuristic_effort;
  int64_t dispfreq;
  std::vector<double> analyticCenter;
  std::vector<double> firstlpsol;
  std::vector<double> rootlpsol;
  double firstlpsolobj;
  HighsBasis firstrootbasis;
  double rootlpsolobj;
  HighsInt numintegercols;
  HighsInt maxTreeSizeLog2;

  HighsCDouble pruned_treeweight;
  double avgrootlpiters;
  double disptime;
  double last_disptime;
  int64_t firstrootlpiters;
  int64_t num_nodes;
  int64_t num_leaves;
  int64_t num_leaves_before_run;
  int64_t num_nodes_before_run;
  int64_t total_repair_lp;
  int64_t total_repair_lp_feasible;
  int64_t total_repair_lp_iterations;
  int64_t total_lp_iterations;
  int64_t heuristic_lp_iterations;
  int64_t sepa_lp_iterations;
  int64_t sb_lp_iterations;
  int64_t total_lp_iterations_before_run;
  int64_t heuristic_lp_iterations_before_run;
  int64_t sepa_lp_iterations_before_run;
  int64_t sb_lp_iterations_before_run;
  int64_t num_disp_lines;

  HighsInt numImprovingSols;
  HighsInt num_consecutive_failed_submips;
  double lower_bound;
  double upper_bound;
  double upper_limit;
  double optimality_limit;
  std::vector<double> incumbent;

  HighsNodeQueue nodequeue;

  // Live hooks for async node-fetch mode (HighsMipSolver::asyncRunNodeFetch).
  // Null unless async mode is running. They let any worker force an early
  // sync epoch when it finds a bound better than the merged global bound,
  // instead of waiting for the periodic epoch budget to drain. The epoch
  // itself (running under the mutex) performs the authoritative merge.
  std::atomic<int64_t>* asyncEpochBudget_ = nullptr;
  std::mutex* asyncNodefetchMutex_ = nullptr;
  std::condition_variable* asyncNodefetchCv_ = nullptr;

  HighsPrimaDualIntegral primal_dual_integral;

  HighsDebugSol debugSolution;

  HighsMipSolverData(HighsMipSolver& mipsolver);

  bool solutionRowFeasible(const std::vector<double>& solution) const;
  HighsModelStatus feasibilityJump();
  HighsModelStatus trivialHeuristics();

  void startAnalyticCenterComputation(
      const highs::parallel::TaskGroup& taskGroup);
  void finishAnalyticCenterComputation(
      const highs::parallel::TaskGroup& taskGroup);

  struct SymmetryDetectionData {
    HighsSymmetryDetection symDetection;
    HighsSymmetries symmetries;
    double detectionTime = 0.0;
  };

  void startSymmetryDetection(const highs::parallel::TaskGroup& taskGroup,
                              std::unique_ptr<SymmetryDetectionData>& symData);
  void finishSymmetryDetection(const highs::parallel::TaskGroup& taskGroup,
                               std::unique_ptr<SymmetryDetectionData>& symData);

  void updatePrimalDualIntegral(const double from_lower_bound,
                                const double to_lower_bound,
                                const double from_upper_bound,
                                const double to_upper_bound,
                                const bool check_bound_change = true,
                                const bool check_prev_data = true);
  double limitsToGap(const double use_lower_bound, const double use_upper_bound,
                     double& lb, double& ub) const;

  double computeNewUpperLimit(double upper_bound, double mip_abs_gap,
                              double mip_rel_gap) const;
  bool moreHeuristicsAllowed() const;
  void removeFixedIndices();
  void init();
  void basisTransfer();
  void checkObjIntegrality();
  void runMipPresolve(const HighsInt presolve_reduction_limit);

  // Independent-components subsolver (SCIP cons_components-style, clean
  // room): solve tiny disconnected (var, constraint) pieces exactly and
  // fix their columns. Called after MIP presolve on the presolved model
  // (and again after a restart, which re-runs presolve); model dimensions
  // are preserved (only bounds tighten) so the postsolve stack stays
  // consistent.
  //
  // Structural information of one detected (var, constraint) block, used
  // for eligibility, debug logging and weak-coupling analysis.
  struct HighsDecompComponent {
    std::vector<HighsInt> cols;
    std::vector<HighsInt> rows;
    HighsInt numInt = 0;       // integer + semi-integer + implicit-integer
    HighsInt numBinary = 0;    // discrete columns with [0, 1] bounds
    HighsInt numContinuous = 0;
    HighsInt numNz = 0;        // internal nonzeros
    HighsInt numObjNz = 0;     // columns with nonzero objective coefficient
  };
  // Accumulated statistics over all decomposition passes of one
  // solveComponents() call. Only informational: solver decisions never
  // depend on these counters.
  struct HighsDecompStats {
    HighsInt numPasses = 0;
    HighsInt numComponents = 0;
    HighsInt numSolved = 0;
    HighsInt numFixed = 0;
    double detectTime = 0.0;
    double solveTime = 0.0;
  };
  void solveComponents();
  // Pure graph analysis: union-find over unfixed columns and rows linked
  // by matrix nonzeros. Fixed columns (lb == ub) are constants, so they
  // are excluded: this lets later passes split blocks that were glued by
  // an already-fixed bridge column. Rows whose columns are all fixed are
  // reported via fixedRows for a consistency check by the caller.
  void detectComponents(const HighsLp& model,
                        std::vector<HighsDecompComponent>& components,
                        std::vector<HighsInt>& fixedRows) const;
  // Solve every eligible component of one pass and fix proven-optimal
  // columns via bounds. Returns false with modelstatus kInfeasible set if
  // a component (or a fully-fixed row) proves global infeasibility.
  bool solveComponentPass(const HighsInt pass, HighsDecompStats& stats);
  // Verify a subsolver solution against the extracted submodel before it
  // is baked into parent bounds: row activities within feastol and
  // integrality within feastol. Never trust kOptimal alone.
  bool verifyComponentSolution(const HighsLp& sublp,
                               const std::vector<double>& subcol) const;
  // Log-only weak-coupling analysis (no solver action): searches for
  // single coupling rows/columns whose removal would split the largest
  // remaining block, and reports whether the structure is a Benders
  // candidate (coupling variables), a dual-decomposition candidate
  // (coupling rows), or heavily coupled.
  void analyzeWeakCoupling(const HighsLp& model,
                           const std::vector<HighsDecompComponent>& components,
                           HighsInt numFixedCols) const;
  // Classical Benders decomposition (implemented in HighsBenders.cpp,
  // same struct, separate translation unit). findBendersSeparator looks
  // for a small column separator whose removal leaves blocks coupled
  // only through the separator; runBenders drives the master/subproblem
  // loop on LP blocks and fixes proven-optimal coupling columns.
  struct HighsBendersCandidate {
    bool valid = false;
    std::string reason;
    std::vector<HighsInt> couplingCols;
    std::vector<std::vector<HighsInt>> blockCols;
    std::vector<std::vector<HighsInt>> blockRows;
    std::vector<HighsInt> masterRows;
    // Per block: true if all columns are continuous (LP subproblem).
    // Non-LP blocks need integer-subproblem support (LP-relaxation cuts
    // plus sub-MIP upper bounds); see mip_benders_integer_subproblems.
    std::vector<char> blockIsLp;
  };
  bool findBendersSeparator(const HighsLp& model,
                            HighsBendersCandidate& cand) const;
  bool runBenders();
  bool verifyBendersSolution(const HighsLp& model,
                             const std::vector<double>& sol) const;
  // Outcome of one presolve-off simplex LP solve (shared by Benders and
  // Lagrangian subproblems so duals/rays refer to the passed submodel).
  struct HighsSubLpResult {
    HighsModelStatus status = HighsModelStatus::kNotset;
    std::vector<double> colSol;
    std::vector<double> rowDual;
    double obj = kHighsInf;
    bool dualValid = false;
  };
  static HighsSubLpResult solveSubLp(const HighsLp& sublp, double timeLimit);
  // Lagrangian decomposition (HighsLagrangian.cpp): row separator whose
  // removal splits the model; coupling rows are dualized (priced) while
  // blocks solve independently. Produces dual bounds (logged) and, when
  // a composed solution is feasible, a verified incumbent injected via
  // the native MIP-start channel. Never fixes variables.
  struct HighsLagCandidate {
    bool valid = false;
    std::string reason;
    std::vector<HighsInt> couplingRows;
    std::vector<std::vector<HighsInt>> blockCols;
    std::vector<std::vector<HighsInt>> blockRows;
  };
  bool findLagSeparator(const HighsLp& model, HighsLagCandidate& cand) const;
  bool runLagrangian();
  void setupDomainPropagation();
  void saveReportMipSolution(const double new_upper_limit = -kHighsInf);
  void checkAddSolution();
  void runSetup();
  double transformNewIntegerFeasibleSolution(
      const std::vector<double>& sol,
      const bool possibly_store_as_new_incumbent = true);
  double percentageInactiveIntegers() const;
  void performRestart();
  bool checkSolution(const std::vector<double>& solution) const;
  std::vector<std::tuple<HighsInt, HighsInt, double>> getInfeasibleRows(
      const std::vector<double>& solution) const;
  bool trySolution(const std::vector<double>& solution,
                   const int solution_source = kSolutionSourceNone);
  bool rootSeparationRound(HighsMipWorker& worker, HighsSeparation& sepa,
                           HighsInt& ncuts, HighsLpRelaxation::Status& status);
  HighsLpRelaxation::Status evaluateRootLp(HighsMipWorker& worker);

  void evaluateRootNode(HighsMipWorker& worker);

  bool addIncumbent(const std::vector<double>& sol, double solobj,
                    const int solution_source,
                    const bool print_display_line = true,
                    const bool is_user_solution = false);

  const std::vector<double>& getSolution() const;

  std::string solutionSourceToString(const int solution_source,
                                     const bool code = true) const;
  void printSolutionSourceKey() const;
  void printDisplayLine(const int solution_source = kSolutionSourceNone);

  void getRow(HighsInt row, HighsInt& rowlen, const HighsInt*& rowinds,
              const double*& rowvals) const {
    HighsInt start = ARstart_[row];
    rowlen = ARstart_[row + 1] - start;
    rowinds = ARindex_.data() + start;
    rowvals = ARvalue_.data() + start;
  }

  bool checkLimits(int64_t nodeOffset = 0) const;
  void limitsToBounds(double& dual_bound, double& primal_bound,
                      double& mip_rel_gap) const;
  void updateLowerBound(double new_lower_bound,
                        const bool check_bound_change = true,
                        const bool check_prev_data = true);
  void setCallbackDataOut(const double mipsolver_objective_value) const;
  bool interruptFromCallbackWithData(const int callback_type,
                                     const double mipsolver_objective_value,
                                     const std::string message = "") const;
  void queryExternalSolution(
      const double mipsolver_objective_value,
      const ExternalMipSolutionQueryOrigin external_solution_query_origin);

  HighsInt terminatorConcurrency() const;
  bool terminatorActive() const { return terminatorConcurrency() > 0; }
  HighsInt terminatorMyInstance() const;
  void terminatorTerminate();
  bool terminatorTerminated() const;
  void terminatorReport() const;

  bool parallelLockActive() const {
    return (parallel_lock && hasMultipleWorkers());
  }

  void updateWorkerEarlyTermination(HighsMipWorker& worker) {
    int64_t current = worker_lp_iterations_stop.load(std::memory_order_relaxed);
    const int64_t candidate = worker.search_ptr_->lpiterations;
    while (candidate < current &&
           !worker_lp_iterations_stop.compare_exchange_weak(
               current, candidate, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    worker.early_termination = true;
  }

  bool hasMultipleWorkers() const { return workers.size() > 1; }

  HighsDomain& getDomain() { return domains[0]; }
  HighsConflictPool& getConflictPool() { return conflictpools[0]; }
  HighsCutPool& getCutPool() { return cutpools[0]; }
  HighsLpRelaxation& getLp() { return lps[0]; }
  HighsPseudocost& getPseudoCost() { return pseudocosts[0]; }
  const HighsDomain& getDomain() const { return domains[0]; }
  const HighsConflictPool& getConflictPool() const { return conflictpools[0]; }
  const HighsCutPool& getCutPool() const { return cutpools[0]; }
  const HighsLpRelaxation& getLp() const { return lps[0]; }
  const HighsPseudocost& getPseudoCost() const { return pseudocosts[0]; }
};

#endif

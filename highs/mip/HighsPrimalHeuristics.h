/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_PRIMAL_HEURISTICS_H_
#define HIGHS_PRIMAL_HEURISTICS_H_

#include <vector>

#include "lp_data/HStruct.h"
#include "lp_data/HighsLp.h"
#include "util/HighsRandom.h"

class HighsMipSolver;
class HighsMipWorker;
class HighsLpRelaxation;

class HighsPrimalHeuristics {
 private:
  const HighsMipSolver& mipsolver;
  std::vector<HighsInt> intcols;
  double successObservations;
  HighsInt numSuccessObservations;
  double infeasObservations;
  HighsInt numInfeasObservations;

  HighsRandom randgen;

 public:
  HighsPrimalHeuristics(HighsMipSolver& mipsolver);

  void setupIntCols();

  bool solveSubMip(HighsMipWorker& worker, const HighsLp& lp,
                   const HighsBasis& basis, double fixingRate,
                   std::vector<double> colLower, std::vector<double> colUpper,
                   HighsInt maxleaves, HighsInt maxnodes, HighsInt stallnodes);

  double determineTargetFixingRate(HighsMipWorker& worker);

  void rootReducedCost(HighsMipWorker& worker);

  void diving(HighsMipWorker& worker);

  bool constraintAwareRounding(HighsMipWorker& worker,
                               const std::vector<double>& relaxationsol);

  void localBranching(HighsMipWorker& worker,
                      const std::vector<double>& relaxationsol);

  // Hamming-ball neighbourhood search (SCIP localbranching-style,
  // clean-room): restrict the neighbourhood of the incumbent and solve
  // the restricted sub-MIP. Binaries enter an L1 ball row (every k-flip
  // combination explored); general integers get a box tightening around
  // the incumbent. Deliberately no auxiliary columns, so sub-MIP
  // dimensions match the parent. Bounded by sub-MIP caps; opt-in.
  void hammingSearch(HighsMipWorker& worker);

  // Proximity search (SCIP proximity-style, clean-room): minimize the
  // binary Hamming distance to the incumbent subject to a strict
  // objective cutoff, and solve the restricted sub-MIP. Forces strict
  // improvement nearby; iterates implicitly as the incumbent improves
  // across node visits. No auxiliary columns; opt-in.
  void proximitySearch(HighsMipWorker& worker);

  void RENS(HighsMipWorker& worker, const std::vector<double>& relaxationsol);

  void RINS(HighsMipWorker& worker, const std::vector<double>& relaxationsol);

  void feasibilityPump(HighsMipWorker& worker);

  void centralRounding(HighsMipWorker& worker);

  void flushStatistics(HighsMipSolver& mipsolver, HighsMipWorker& worker);

  bool tryRoundedPoint(HighsMipWorker& worker, const std::vector<double>& point,
                       const int solution_source);

  bool linesearchRounding(HighsMipWorker& worker,
                          const std::vector<double>& point1,
                          const std::vector<double>& point2,
                          const int solution_source);

  void randomizedRounding(HighsMipWorker& worker,
                          const std::vector<double>& relaxationsol);

  void shifting(HighsMipWorker& worker,
                const std::vector<double>& relaxationsol);

  void ziRound(HighsMipWorker& worker,
               const std::vector<double>& relaxationsol);

  bool addIncumbent(const std::vector<double>& sol, double solobj,
                    const int solution_source, HighsMipWorker& worker);

  bool trySolution(const std::vector<double>& solution,
                   const int solution_source, HighsMipWorker& worker);

  HighsInt getNumSuccessObservations(HighsMipWorker& worker) const;

  HighsInt getNumInfeasObservations(HighsMipWorker& worker) const;

  double getSuccessObservations(HighsMipWorker& worker) const;

  double getInfeasObservations(HighsMipWorker& worker) const;

  HighsInt getHeuristicRandom(const HighsInt sup) {
    return randgen.integer(sup);
  }
};

#endif

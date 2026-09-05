#include "HCheckConfig.h"
#include "Highs.h"
#include "SpecialLps.h"
#include "catch.hpp"

const bool dev_run = false;
const double double_equal_tolerance = 1e-5;

bool objectiveOk(const double optimal_objective,
                 const double require_optimal_objective,
                 const bool dev_run = false);

void solve(Highs& highs, std::string presolve,
           const HighsModelStatus require_model_status,
           const double require_optimal_objective = 0,
           const double require_iteration_count = -1);
void distillationMIP(Highs& highs);
void rowlessMIP(Highs& highs);
void rowlessMIP1(Highs& highs);
void rowlessMIP2(Highs& highs);

TEST_CASE("MIP-distillation", "[highs_test_mip_solver]") {
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  distillationMIP(highs);

  highs.resetGlobalScheduler(true);
}

// Fails but the cases work separately in
// MIP-rowless-1 and
// MIP-rowless-2 below
// TEST_CASE("MIP-rowless", "[highs_test_mip_solver]") {
//   Highs highs;
//   if (!dev_run) highs.setOptionValue("output_flag", false);
//   rowlessMIP(highs);
// }

TEST_CASE("MIP-rowless-1", "[highs_test_mip_solver]") {
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  rowlessMIP1(highs);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-rowless-2", "[highs_test_mip_solver]") {
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  rowlessMIP2(highs);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-solution-limit", "[highs_test_mip_solver]") {
  std::string filename;
  filename = std::string(HIGHS_DIR) + "/check/instances/rgn.mps";

  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  highs.readModel(filename);

  highs.setOptionValue("presolve", kHighsOffString);
  if (dev_run) highs.setOptionValue("log_dev_level", 1);

  // Test for kSolutionLimit with mip_max_nodes
  highs.setOptionValue("mip_max_nodes", 0);
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kSolutionLimit);
  highs.setOptionValue("mip_max_nodes", kHighsIInf);
  highs.clearSolver();

  // Test for kSolutionLimit with mip_max_leaves
  highs.setOptionValue("mip_max_leaves", 0);
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kSolutionLimit);
  highs.setOptionValue("mip_max_leaves", kHighsIInf);
  highs.clearSolver();

  // Test for kSolutionLimit with mip_max_improving_sols
  highs.setOptionValue("mip_max_improving_sols", 1);
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kSolutionLimit);
  highs.setOptionValue("mip_max_improving_sols", kHighsIInf);
  highs.clearSolver();

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-integrality", "[highs_test_mip_solver]") {
  std::string filename;
  filename = std::string(HIGHS_DIR) + "/check/instances/avgas.mps";

  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  highs.readModel(filename);
  highs.run();
  highs.readModel(filename);
  const HighsLp& lp = highs.getLp();
  const HighsInfo& info = highs.getInfo();
  vector<HighsVarType> integrality;
  integrality.resize(lp.num_col_);
  HighsInt from_col0 = 0;
  HighsInt to_col0 = 2;
  HighsInt from_col1 = 5;
  HighsInt to_col1 = 7;
  HighsInt num_set_entries = 6;
  vector<HighsInt> set;
  set.push_back(0);
  set.push_back(7);
  set.push_back(1);
  set.push_back(5);
  set.push_back(2);
  set.push_back(6);
  vector<HighsInt> mask;
  mask.assign(lp.num_col_, 0);
  for (HighsInt ix = 0; ix < num_set_entries; ix++) {
    HighsInt iCol = set[ix];
    mask[iCol] = 1;
    integrality[ix] = HighsVarType::kInteger;
  }
  REQUIRE(highs.changeColsIntegrality(from_col0, to_col0, integrality.data()) ==
          HighsStatus::kOk);
  REQUIRE(highs.changeColsIntegrality(from_col1, to_col1, integrality.data()) ==
          HighsStatus::kOk);
  if (dev_run) {
    highs.setOptionValue("log_dev_level", 3);
  } else {
    highs.setOptionValue("output_flag", false);
  }
  if (dev_run) highs.writeModel("");
  highs.run();
  if (dev_run) highs.writeSolution("", kSolutionStylePretty);
  double optimal_objective = info.objective_function_value;
  if (dev_run) printf("Objective = %g\n", optimal_objective);

  // mip_node_count is always int64_t, so the following should be an
  // error depending on whether HIGHSINT64 is set
  HighsInt mip_node_count_int;
  HighsStatus required_return_status = HighsStatus::kError;
#ifdef HIGHSINT64
  required_return_status = HighsStatus::kOk;
#endif
  REQUIRE(highs.getInfoValue("mip_node_count", mip_node_count_int) ==
          required_return_status);
  int64_t mip_node_count;
  REQUIRE(highs.getInfoValue("mip_gap", mip_node_count) == HighsStatus::kError);
  REQUIRE(highs.getInfoValue("mip_node_count", mip_node_count) ==
          HighsStatus::kOk);
  REQUIRE(mip_node_count == 1);

  highs.clearModel();
  if (!dev_run) highs.setOptionValue("output_flag", false);
  highs.readModel(filename);
  REQUIRE(highs.changeColsIntegrality(num_set_entries, set.data(),
                                      integrality.data()) == HighsStatus::kOk);
  if (dev_run) highs.writeModel("");
  highs.run();
  if (dev_run) highs.writeSolution("", kSolutionStylePretty);
  REQUIRE(info.objective_function_value == optimal_objective);

  integrality.assign(lp.num_col_, HighsVarType::kContinuous);
  for (HighsInt ix = 0; ix < num_set_entries; ix++) {
    HighsInt iCol = set[ix];
    integrality[iCol] = HighsVarType::kInteger;
  }

  highs.clearModel();
  if (!dev_run) highs.setOptionValue("output_flag", false);
  highs.readModel(filename);
  REQUIRE(highs.changeColsIntegrality(mask.data(), integrality.data()) ==
          HighsStatus::kOk);
  if (dev_run) highs.writeModel("");
  highs.run();
  if (dev_run) highs.writeSolution("", kSolutionStylePretty);
  if (dev_run) highs.writeSolution("", kSolutionStyleRaw);
  REQUIRE(info.objective_function_value == optimal_objective);

  REQUIRE(info.mip_node_count == 1);
  REQUIRE(fabs(info.mip_dual_bound + 6) < double_equal_tolerance);
  REQUIRE(std::fabs(info.mip_gap) < 1e-12);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-clear-integrality", "[highs_test_mip_solver]") {
  SpecialLps special_lps;
  HighsLp lp;
  HighsModelStatus require_model_status;
  double optimal_objective;
  special_lps.distillationMip(lp, require_model_status, optimal_objective);
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.passModel(lp);
  REQUIRE(highs.getLp().integrality_.size() > 0);
  highs.clearIntegrality();
  REQUIRE(highs.getLp().integrality_.size() == 0);
}

TEST_CASE("MIP-nmck", "[highs_test_mip_solver]") {
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 2;
  lp.col_cost_ = {-3, -2, -1};
  lp.col_lower_ = {0, 0, 0};
  lp.col_upper_ = {inf, inf, 1};
  lp.row_lower_ = {-inf, 12};
  lp.row_upper_ = {7, 12};
  lp.a_matrix_.start_ = {0, 2, 4, 6};
  lp.a_matrix_.index_ = {0, 1, 0, 1, 0, 1};
  lp.a_matrix_.value_ = {1, 4, 1, 2, 1, 1};
  lp.integrality_ = {HighsVarType::kContinuous, HighsVarType::kContinuous,
                     HighsVarType::kInteger};
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  highs.setOptionValue("highs_debug_level", kHighsDebugLevelCheap);
  if (dev_run) highs.setOptionValue("log_dev_level", 2);
  HighsStatus return_status = highs.run();
  REQUIRE(return_status == HighsStatus::kOk);
  if (dev_run) highs.writeInfo("");
  const HighsInfo& info = highs.getInfo();
  REQUIRE(info.num_primal_infeasibilities == 0);
  REQUIRE(info.max_primal_infeasibility == 0);
  REQUIRE(info.sum_primal_infeasibilities == 0);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-maximize", "[highs_test_mip_solver]") {
  SpecialLps special_lps;
  HighsLp lp;
  HighsModelStatus require_model_status;
  double optimal_objective;
  special_lps.distillationMip(lp, require_model_status, optimal_objective);
  // Add an offset to make sure this is handled correctly
  double offset = -20;
  lp.offset_ = offset;
  optimal_objective += offset;
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  const HighsInfo& info = highs.getInfo();
  const HighsOptions& options = highs.getOptions();
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(std::abs(info.objective_function_value - optimal_objective) <
          double_equal_tolerance);
  REQUIRE(std::abs(info.objective_function_value - info.mip_dual_bound) <=
          options.mip_abs_gap);
  REQUIRE(std::abs(info.mip_gap) <= options.mip_rel_gap);

  // Turn the problem into a maximization
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) lp.col_cost_[iCol] *= -1;
  lp.offset_ *= -1;
  optimal_objective *= -1;
  lp.sense_ = ObjSense::kMaximize;
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(std::abs(info.objective_function_value - optimal_objective) <
          double_equal_tolerance);
  REQUIRE(std::abs(info.objective_function_value - info.mip_dual_bound) <=
          options.mip_abs_gap);
  REQUIRE(std::abs(info.mip_gap) <= options.mip_rel_gap);

  highs.setOptionValue("solve_relaxation", true);
  optimal_objective = -11.2;
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(std::abs(info.objective_function_value - optimal_objective) <
          double_equal_tolerance);
  highs.setOptionValue("solve_relaxation", false);

  // Now test with a larger problem
  const bool use_avgas = true;
  const std::string model = use_avgas ? "avgas" : "dcmulti";
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/" + model + ".mps";
  highs.readModel(filename);
  optimal_objective = use_avgas ? -6.0 : 188182;
  offset = 0;  // 5;
  optimal_objective += offset;
  lp = highs.getLp();
  lp.offset_ = offset;
  // Turn the model into a maximization MIP
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    lp.col_cost_[iCol] *= -1;
    if (use_avgas) lp.integrality_.push_back(HighsVarType::kInteger);
  }
  lp.offset_ *= -1;
  optimal_objective *= -1;
  lp.sense_ = ObjSense::kMaximize;
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("mip_rel_gap", 0.0);

  REQUIRE(highs.run() == HighsStatus::kOk);
  if (dev_run) {
    printf("optimal_objective =             %11.4g\n", optimal_objective);
    printf("info.objective_function_value = %11.4g\n",
           info.objective_function_value);
    printf("info.mip_dual_bound =           %11.4g\n", info.mip_dual_bound);
    printf("info.mip_gap =                  %11.4g\n", info.mip_gap);
  }
  REQUIRE(std::abs(info.objective_function_value - optimal_objective) <
          double_equal_tolerance);
  REQUIRE(std::abs(info.objective_function_value - info.mip_dual_bound) <=
          options.mip_abs_gap);
  REQUIRE(std::abs(info.mip_gap) <= options.mip_rel_gap);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-unbounded", "[highs_test_mip_solver]") {
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  HighsLp lp;
  HighsStatus return_status;
  HighsModelStatus model_status;
  // One-variable unbounded MIP from SciPy HiGHS MIP wrapper #28
  lp.num_col_ = 1;
  lp.num_row_ = 0;
  lp.col_cost_ = {-1};
  lp.col_lower_ = {0};
  lp.col_upper_ = {inf};
  lp.integrality_ = {HighsVarType::kInteger};

  bool use_presolve = false;
  HighsModelStatus require_model_status;
  for (HighsInt k = 0; k < 2; k++) {
    if (use_presolve) {
      // With use_presolve = true, MIP solver returns
      // HighsModelStatus::kUnboundedOrInfeasible from presolve
      highs.setOptionValue("presolve", kHighsOnString);
      require_model_status = HighsModelStatus::kUnboundedOrInfeasible;
    } else {
      // With use_presolve = false, MIP solver returns
      // HighsModelStatus::kUnbounded, because the all-zeros trivial
      // heuristic finds a feasible point
      //
      // Feasibility jump appears to find one before the all-zeros
      // trivial heuristic
      highs.setOptionValue("presolve", kHighsOffString);
      require_model_status = HighsModelStatus::kUnbounded;
    }
    return_status = highs.passModel(lp);
    REQUIRE(return_status == HighsStatus::kOk);

    return_status = highs.run();
    REQUIRE(return_status == HighsStatus::kOk);

    model_status = highs.getModelStatus();
    REQUIRE(model_status == require_model_status);

    // Second time through loop is with presolve
    use_presolve = true;
  }
  // Two-variable problem that is also primal unbounded as an LP, but
  // primal infeasible as a MIP.
  //
  // min -x subject to x+2y>=1, x>=0; 1/4 <= y <= 3/4; y\in{0,1}
  //
  // First the LP - unbounded
  lp.clear();
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {-1, 0};
  lp.col_lower_ = {0, 0.25};
  lp.col_upper_ = {inf, 0.75};
  lp.row_lower_ = {1};
  lp.row_upper_ = {inf};
  lp.a_matrix_.start_ = {0, 2};
  lp.a_matrix_.index_ = {0, 1};
  lp.a_matrix_.value_ = {1, 2};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;

  use_presolve = false;
  for (HighsInt k = 0; k < 2; k++) {
    if (use_presolve) {
      // With use_presolve = true, LP solver returns
      // HighsModelStatus::kUnbounded because it solves the LP after
      // presolve has returned
      highs.setOptionValue("presolve", kHighsOnString);
      require_model_status = HighsModelStatus::kUnbounded;
    } else {
      // With use_presolve = false, LP solver returns
      // HighsModelStatus::kUnbounded
      highs.setOptionValue("presolve", kHighsOffString);
      require_model_status = HighsModelStatus::kUnbounded;
    }

    return_status = highs.passModel(lp);
    REQUIRE(return_status == HighsStatus::kOk);

    return_status = highs.run();
    REQUIRE(return_status == HighsStatus::kOk);

    model_status = highs.getModelStatus();
    REQUIRE(model_status == require_model_status);

    // Second time through loop is with presolve
    use_presolve = true;
  }

  // Now as a MIP - infeasible
  lp.integrality_ = {HighsVarType::kContinuous, HighsVarType::kInteger};
  // With(out) presolve, Highs::infeasibleBoundsOk() performs inward
  // integer rounding of [0.25, 0.75] to [1, 0] so identifes
  // infeasiblility. Hence MIP solver returns
  // HighsModelStatus::kInfeasible

  return_status = highs.passModel(lp);
  REQUIRE(return_status == HighsStatus::kOk);

  return_status = highs.run();
  REQUIRE(return_status == HighsStatus::kOk);

  model_status = highs.getModelStatus();
  REQUIRE(model_status == HighsModelStatus::kInfeasible);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-od", "[highs_test_mip_solver]") {
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  HighsLp lp;
  lp.num_col_ = 1;
  lp.num_row_ = 0;
  lp.col_cost_ = {-2};
  lp.col_lower_ = {-inf};
  lp.col_upper_ = {1.5};
  lp.integrality_ = {HighsVarType::kInteger};
  double required_objective_value = -2;
  double required_x0_value = 1;

  const HighsInfo& info = highs.getInfo();
  const HighsSolution& solution = highs.getSolution();

  HighsStatus return_status = highs.passModel(lp);
  REQUIRE(return_status == HighsStatus::kOk);

  if (dev_run) {
    printf("One variable unconstrained MIP: model\n");
    highs.writeModel("");
  }

  return_status = highs.run();
  REQUIRE(return_status == HighsStatus::kOk);

  const HighsInt style = kSolutionStylePretty;
  if (dev_run) {
    printf("One variable unconstrained MIP: solution\n");
    highs.writeSolution("", style);
  }

  HighsModelStatus model_status = highs.getModelStatus();

  REQUIRE(model_status == HighsModelStatus::kOptimal);
  REQUIRE(fabs(info.objective_function_value - required_objective_value) <
          double_equal_tolerance);
  REQUIRE(fabs(solution.col_value[0] - required_x0_value) <
          double_equal_tolerance);

  highs.changeColBounds(0, -2, 2);

  if (dev_run) {
    printf("After changing bounds: model\n");
    highs.writeModel("");
  }

  return_status = highs.run();
  REQUIRE(return_status == HighsStatus::kOk);

  model_status = highs.getModelStatus();

  if (dev_run) {
    printf("After changing bounds: solution\n");
    highs.writeSolution("", style);
  }

  required_objective_value = -4;
  required_x0_value = 2;
  REQUIRE(model_status == HighsModelStatus::kOptimal);
  REQUIRE(fabs(info.objective_function_value - required_objective_value) <
          double_equal_tolerance);
  REQUIRE(fabs(solution.col_value[0] - required_x0_value) <
          double_equal_tolerance);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-infeasible-start", "[highs_test_mip_solver]") {
  HighsSolution sol;
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  const HighsModelStatus& model_status = highs.getModelStatus();
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 2;
  lp.col_cost_ = {0, 0};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1.5, 1.5};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  const double rhs = 4.0;
  const double delta = 0.99;
  lp.row_lower_ = {rhs - delta, rhs + delta};
  lp.row_upper_ = {rhs - delta, rhs + delta};
  lp.a_matrix_.start_ = {0, 2, 4};
  lp.a_matrix_.index_ = {0, 1, 0, 1};
  lp.a_matrix_.value_ = {1, 2, 2, 1};

  highs.passModel(lp);

  sol.col_value = {1, 1};
  highs.setSolution(sol);
  //  REQUIRE(highs.setOptionValue("presolve", kHighsOffString) ==
  //  HighsStatus::kOk);
  highs.run();
  REQUIRE(model_status == HighsModelStatus::kInfeasible);

  // Stefan's example
  std::string filename;
  filename = std::string(HIGHS_DIR) + "/check/instances/infeasible-mip1.mps";

  highs.readModel(filename);
  sol.col_value = {75, 0, 275, 300, 300, 0, 0, 0, 50, 0, 0,
                   1,  0, 1,   1,   1,   0, 0, 0, 1,  0, 0};
  highs.setSolution(sol);
  REQUIRE(highs.setOptionValue("presolve", kHighsOffString) ==
          HighsStatus::kOk);
  highs.run();
  REQUIRE(model_status == HighsModelStatus::kInfeasible);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("get-integrality", "[highs_test_mip_solver]") {}

TEST_CASE("MIP-bounds", "[highs_test_mip_solver]") {
  const std::string test_name = Catch::getResultCapture().getCurrentTestName();
  const std::string test_mps = test_name + ".mps";
  // Introduced due to #1325 observing that LI and UI are needed
  HighsLp lp;
  lp.num_col_ = 6;
  lp.num_row_ = 3;
  lp.col_cost_ = {1, 1, 1, 2, 2, 2};
  lp.col_lower_ = {0, 0, 0, 0, 0, 0};
  lp.col_upper_ = {kHighsInf, kHighsInf, kHighsInf,
                   kHighsInf, kHighsInf, kHighsInf};
  lp.integrality_ = {HighsVarType::kInteger,    HighsVarType::kInteger,
                     HighsVarType::kInteger,    HighsVarType::kContinuous,
                     HighsVarType::kContinuous, HighsVarType::kContinuous};
  const double rhs = 10.99;
  lp.row_lower_ = {rhs, rhs, rhs};
  lp.row_upper_ = {kHighsInf, kHighsInf, kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.num_col_ = lp.num_col_;
  lp.a_matrix_.num_row_ = lp.num_row_;
  lp.a_matrix_.start_ = {0, 1, 2, 3, 4, 5, 6};
  lp.a_matrix_.index_ = {0, 1, 2, 0, 1, 2};
  lp.a_matrix_.value_ = {1, 1, 1, 1, 1, 1};
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.passModel(lp);
  highs.run();
  const double obj0 = highs.getObjectiveValue();
  if (dev_run) printf("Optimum at first run: %g\n", obj0);
  // now write out to MPS and load again
  highs.writeModel(test_mps);
  highs.readModel(test_mps);
  highs.run();
  const double obj1 = highs.getObjectiveValue();
  if (dev_run)
    printf("Optimum at second run (after writing and loading again): %g\n",
           obj1);
  REQUIRE(obj0 == obj1);
  std::remove(test_mps.c_str());

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-get-saved-solutions", "[highs_test_mip_solver]") {
  const std::string test_name = Catch::getResultCapture().getCurrentTestName();
  const std::string solution_file = test_name + ".sol";
  const std::string model = "flugpl";
  const std::string model_file =
      std::string(HIGHS_DIR) + "/check/instances/" + model + ".mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("mip_improving_solution_save", true);
  highs.setOptionValue("mip_improving_solution_report_sparse", true);
  highs.setOptionValue("mip_improving_solution_file", solution_file);
  highs.readModel(model_file);
  highs.run();
  const std::vector<HighsObjectiveSolution> saved_objective_and_solution =
      highs.getSavedMipSolutions();
  const HighsInt num_saved_solution = saved_objective_and_solution.size();
  REQUIRE(num_saved_solution > 0);
  const HighsInt last_saved_solution = num_saved_solution - 1;
  REQUIRE(saved_objective_and_solution[last_saved_solution].objective ==
          highs.getInfo().objective_function_value);
  for (HighsInt iCol = 0; iCol < highs.getLp().num_col_; iCol++)
    REQUIRE(saved_objective_and_solution[last_saved_solution].col_value[iCol] ==
            highs.getSolution().col_value[iCol]);
  std::remove(solution_file.c_str());

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-objective-target", "[highs_test_mip_solver]") {
  const double egout_optimal_objective = 568.1007;
  const double egout_objective_target = 610;
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/egout.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("objective_target", egout_objective_target);
  highs.readModel(filename);
  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kObjectiveTarget);
  REQUIRE(highs.getInfo().objective_function_value > egout_optimal_objective);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-max-offset-test", "[highs_test_mip_solver]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/egout.mps";
  const double offset = 100;
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.readModel(filename);
  highs.run();
  const double og_optimal_objective = highs.getInfo().objective_function_value;
  HighsLp lp = highs.getLp();
  lp.offset_ = offset;
  highs.passModel(lp);
  highs.run();
  const double offset_optimal_objective =
      highs.getInfo().objective_function_value;
  REQUIRE(objectiveOk(offset + og_optimal_objective, offset_optimal_objective,
                      dev_run));

  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) lp.col_cost_[iCol] *= -1;
  lp.offset_ *= -1;
  lp.sense_ = ObjSense::kMaximize;
  highs.passModel(lp);
  highs.run();
  const double max_offset_optimal_objective =
      highs.getInfo().objective_function_value;
  REQUIRE(objectiveOk(max_offset_optimal_objective, -offset_optimal_objective,
                      dev_run));

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-get-saved-solutions-presolve", "[highs_test_mip_solver]") {
  const std::string test_name = Catch::getResultCapture().getCurrentTestName();
  const std::string solution_file = test_name + ".sol";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_improving_solution_save", true);
  highs.setOptionValue("mip_improving_solution_report_sparse", true);
  highs.setOptionValue("mip_improving_solution_file", solution_file);
  // #1724: Add row to the example so that solution is non-zero
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {1, 1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 1};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  lp.row_lower_ = {1};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.num_col_ = 2;
  lp.a_matrix_.num_row_ = 1;
  lp.a_matrix_.start_ = {0, 1, 1};
  lp.a_matrix_.index_ = {0};
  lp.a_matrix_.value_ = {1};
  highs.passModel(lp);
  highs.run();
  const std::vector<HighsObjectiveSolution> saved_objective_and_solution =
      highs.getSavedMipSolutions();
  const HighsInt num_saved_solution = saved_objective_and_solution.size();
  REQUIRE(num_saved_solution == 1);
  const HighsInt last_saved_solution = num_saved_solution - 1;
  REQUIRE(saved_objective_and_solution[last_saved_solution].objective ==
          highs.getInfo().objective_function_value);
  for (HighsInt iCol = 0; iCol < highs.getLp().num_col_; iCol++)
    REQUIRE(saved_objective_and_solution[last_saved_solution].col_value[iCol] ==
            highs.getSolution().col_value[iCol]);
  std::remove(solution_file.c_str());

  highs.resetGlobalScheduler(true);
}

TEST_CASE("IP-infeasible-unbounded", "[highs_test_mip_solver]") {
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  double delta = 0.2;
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 0;
  lp.col_cost_ = {-1, 0};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  highs.setOptionValue("presolve", kHighsOffString);
  for (HighsInt k = 0; k < 2; k++) {
    for (HighsInt l = 0; l < 2; l++) {
      if (l == 0) {
        // Infeasible
        lp.col_lower_ = {0, delta};
        lp.col_upper_ = {kHighsInf, 1 - delta};
      } else {
        // Unbounded
        lp.col_lower_ = {0, -delta};
        lp.col_upper_ = {kHighsInf, 1 + delta};
      }
      // Solve
      highs.passModel(lp);
      highs.run();
      HighsModelStatus required_model_status;
      if (k == 0) {
        // Presolve off
        if (l == 0) {
          // MIP solver proves infeasiblilty
          required_model_status = HighsModelStatus::kInfeasible;
        } else {
          // Relaxation is unbounded, but origin is feasible
          required_model_status = HighsModelStatus::kUnbounded;
        }
      } else {
        // Presolve on
        if (l == 0) {
          // Inward integer rounding proves infeasiblilty
          required_model_status = HighsModelStatus::kInfeasible;
        } else {
          // Presolve identifies primal infeasible or unbounded
          required_model_status = HighsModelStatus::kUnboundedOrInfeasible;
        }
      }
      if (dev_run)
        printf(
            "For k = %d and l = %d, original bounds on col 1 are [%g, %g]: "
            "model status is \"%s\" and required status is \"%s\"\n",
            int(k), int(l), lp.col_lower_[1], lp.col_upper_[1],
            highs.modelStatusToString(highs.getModelStatus()).c_str(),
            highs.modelStatusToString(required_model_status).c_str());
      REQUIRE(highs.getModelStatus() == required_model_status);
    }
    highs.setOptionValue("presolve", kHighsOnString);
  }

  highs.resetGlobalScheduler(true);
}

TEST_CASE("IP-with-fract-bounds-no-presolve", "[highs_test_mip_solver]") {
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  // No presolve
  highs.setOptionValue("presolve", kHighsOffString);

  // IP without constraints and fractional bounds on variables
  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 0;
  lp.col_cost_ = {1, -2, 3};
  lp.col_lower_ = {2.5, 2.5, 2.5};
  lp.col_upper_ = {6.5, 5.5, 7.5};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger,
                     HighsVarType::kInteger};

  // Solve
  highs.passModel(lp);
  highs.run();

  // Check status and optimal objective value
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(objectiveOk(highs.getInfo().objective_function_value, 2.0, dev_run));

  // Fix an integer variable to a fractional value
  lp.col_upper_[0] = 2.5;

  // Solve again
  highs.passModel(lp);
  highs.run();

  // Infeasible
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kInfeasible);

  highs.resetGlobalScheduler(true);
}

/*
TEST_CASE("MIP-2084", "[highs_test_mip_solver]") {
// To be used to debug #2084
  Highs h;
  // No presolve
  h.setOptionValue("output_flag", dev_run);

  // Minimize
  //   3x + y
  // Subject to
  //   47x + 19y = 10000000002226
  //   23x + 57y = 10000000013254
  // General
  //   x y
  // End

  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 2;
  lp.col_cost_ = {3, 1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {kHighsInf, kHighsInf};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  lp.row_lower_ = {10000000002226, 10000000013254};
  lp.row_upper_ = {10000000002226, 10000000013254};
  lp.a_matrix_.start_ = {0, 2, 4};
  lp.a_matrix_.index_ = {0, 1, 0, 1};
  lp.a_matrix_.value_ = {47, 23, 19, 57};

  // Solve
  h.passModel(lp);
  h.setOptionValue("presolve", "off");
  h.run();
  HighsModelStatus require_model_status = h.getModelStatus();
  if (dev_run)
    printf("Solution is [%24.18g, %24.18g] with status %s\n",
           h.getSolution().col_value[0], h.getSolution().col_value[1],
           h.modelStatusToString(require_model_status).c_str());

  h.clearSolver();

  h.setOptionValue("presolve", "on");
  h.run();
  HighsModelStatus model_status = h.getModelStatus();
  if (dev_run)
    printf("Solution is [%24.18g, %24.18g] with status %s\n",
           h.getSolution().col_value[0], h.getSolution().col_value[1],
           h.modelStatusToString(model_status).c_str());
  REQUIRE(model_status == require_model_status);
}
*/

bool objectiveOk(const double optimal_objective,
                 const double require_optimal_objective, const bool dev_run) {
  double error = std::fabs(optimal_objective - require_optimal_objective) /
                 std::max(1.0, std::fabs(require_optimal_objective));
  bool error_ok = error < 1e-10;
  if (!error_ok && dev_run)
    printf("Objective is %g but require %g (error %g)\n", optimal_objective,
           require_optimal_objective, error);
  return error_ok;
}

void solve(Highs& highs, std::string presolve,
           const HighsModelStatus require_model_status,
           const double require_optimal_objective,
           const double require_iteration_count) {
  if (!dev_run) highs.setOptionValue("output_flag", false);
  const HighsInfo& info = highs.getInfo();
  REQUIRE(highs.setOptionValue("presolve", presolve) == HighsStatus::kOk);

  REQUIRE(highs.setBasis() == HighsStatus::kOk);

  REQUIRE(highs.run() == HighsStatus::kOk);

  REQUIRE(highs.getModelStatus() == require_model_status);

  if (require_model_status == HighsModelStatus::kOptimal) {
    REQUIRE(objectiveOk(info.objective_function_value,
                        require_optimal_objective, dev_run));
  }
  REQUIRE(highs.resetOptions() == HighsStatus::kOk);

  highs.resetGlobalScheduler(true);
}

void distillationMIP(Highs& highs) {
  SpecialLps special_lps;
  HighsLp lp;
  HighsModelStatus require_model_status;
  double optimal_objective;
  special_lps.distillationMip(lp, require_model_status, optimal_objective);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  // Presolve doesn't reduce the LP
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

void rowlessMIP(Highs& highs) {
  HighsLp lp;
  HighsModelStatus require_model_status;
  double optimal_objective;
  lp.num_col_ = 2;
  lp.num_row_ = 0;
  lp.col_cost_ = {1, -1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 1};
  lp.a_matrix_.start_ = {0, 0, 0};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.sense_ = ObjSense::kMinimize;
  lp.offset_ = 0;
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  require_model_status = HighsModelStatus::kOptimal;
  optimal_objective = -1.0;
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  // Presolve reduces the LP to empty
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
  solve(highs, kHighsOffString, require_model_status, optimal_objective);
}

void rowlessMIP1(Highs& highs) {
  HighsLp lp;
  HighsModelStatus require_model_status;
  double optimal_objective;
  lp.num_col_ = 2;
  lp.num_row_ = 0;
  lp.col_cost_ = {1, -1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 1};
  lp.a_matrix_.start_ = {0, 0, 0};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.sense_ = ObjSense::kMinimize;
  lp.offset_ = 0;
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  require_model_status = HighsModelStatus::kOptimal;
  optimal_objective = -1.0;
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  // Presolve reduces the LP to empty
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
  // solve(highs, kHighsOffString, require_model_status, optimal_objective);
}

void rowlessMIP2(Highs& highs) {
  HighsLp lp;
  HighsModelStatus require_model_status;
  double optimal_objective;
  lp.num_col_ = 2;
  lp.num_row_ = 0;
  lp.col_cost_ = {1, -1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 1};
  lp.a_matrix_.start_ = {0, 0, 0};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.sense_ = ObjSense::kMinimize;
  lp.offset_ = 0;
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  require_model_status = HighsModelStatus::kOptimal;
  optimal_objective = -1.0;
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  // Presolve reduces the LP to empty
  // solve(highs, kHighsOnString, require_model_status, optimal_objective);
  solve(highs, kHighsOffString, require_model_status, optimal_objective);
}

TEST_CASE("issue-2122", "[highs_test_mip_solver]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/2122.lp";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.readModel(filename);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = -187612.944194;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("issue-2171", "[highs_test_mip_solver]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/2171.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.readModel(filename);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = -22375.7585461;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("issue-2204", "[highs_test_mip_solver]") {
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/issue-2204.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.readModel(filename);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = 6.0;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("ZI Round and Shifting Heuristics", "[highs_test_mip_solver]") {
  std::string model_file = std::string(HIGHS_DIR) + "/check/instances/rgn.mps";

  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  // Enable both heuristics
  highs.setOptionValue("mip_heuristic_run_zi_round", true);
  highs.setOptionValue("mip_heuristic_run_shifting", true);
  highs.readModel(model_file);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = 82.19999924;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("issue-2290", "[highs_test_mip_solver]") {
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/issue-2290.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.readModel(filename);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = -1.6666666666;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("issue-2409", "[highs_test_mip_solver]") {
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 2;
  lp.col_cost_ = {-1, 1};
  lp.col_lower_ = {-kHighsInf, -kHighsInf};
  lp.col_upper_ = {kHighsInf, kHighsInf};
  lp.row_lower_ = {0.1, 0.1};
  lp.row_upper_ = {kHighsInf, kHighsInf};
  lp.a_matrix_.start_ = {0, 2, 4};
  lp.a_matrix_.index_ = {0, 1, 0, 1};
  lp.a_matrix_.value_ = {-1, 1, 1, 1};
  lp.integrality_ = {HighsVarType::kContinuous, HighsVarType::kInteger};
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = 0.1;
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  if (dev_run) printf("Testing that presolve reduces the problem to empty\n");
  REQUIRE(highs.presolve() == HighsStatus::kOk);
  REQUIRE(highs.getModelPresolveStatus() ==
          HighsPresolveStatus::kReducedToEmpty);

  if (dev_run)
    printf(
        "\nTesting that with presolve the correct optimal objective is "
        "found\n");
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
  highs.clearSolver();
  if (dev_run)
    printf(
        "\nTesting that without presolve the correct optimal objective is "
        "found\n");
  solve(highs, kHighsOffString, require_model_status, optimal_objective);
}

TEST_CASE("issue-2432", "[highs_test_mip_solver]") {
  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 3;
  lp.col_cost_ = {-93, 25, 17};
  lp.col_lower_ = {-100, -100, -100};
  lp.col_upper_ = {120, 10, 0};
  lp.row_lower_ = {3994.5, -4878.3, -4930};
  lp.row_upper_ = {kHighsInf, kHighsInf, kHighsInf};
  lp.a_matrix_.start_ = {0, 3, 6, 9};
  lp.a_matrix_.index_ = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  lp.a_matrix_.value_ = {-89, -0.1, -8.6, -40.7, 77.2, -6.5, -12, -23.7, 72.78};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous,
                     HighsVarType::kInteger};
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = -3777.57124352;
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  if (dev_run) printf("Testing that presolve reduces the problem\n");
  REQUIRE(highs.presolve() == HighsStatus::kOk);
  REQUIRE(highs.getModelPresolveStatus() == HighsPresolveStatus::kReduced);

  if (dev_run)
    printf(
        "\nTesting that with presolve the correct optimal objective is "
        "found\n");
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
  highs.clearSolver();
  if (dev_run)
    printf(
        "\nTesting that without presolve the correct optimal objective is "
        "found\n");
  solve(highs, kHighsOffString, require_model_status, optimal_objective);
}

TEST_CASE("mip-lp-solver-string", "[highs_test_mip_solver]") {
  Highs h;
  h.setOptionValue("output_flag", dev_run);
  REQUIRE(h.setOptionValue(kMipLpSolverString, "fred") == HighsStatus::kError);
  REQUIRE(h.setOptionValue(kMipLpSolverString, kHighsChooseString) ==
          HighsStatus::kOk);
  REQUIRE(h.setOptionValue(kMipLpSolverString, kSimplexString) ==
          HighsStatus::kOk);
  REQUIRE(h.setOptionValue(kMipLpSolverString, kIpmString) == HighsStatus::kOk);

#ifdef HIPO
  REQUIRE(h.setOptionValue(kMipLpSolverString, kHipoString) ==
          HighsStatus::kOk);
#else
  REQUIRE(h.setOptionValue(kMipLpSolverString, kHipoString) ==
          HighsStatus::kError);
#endif

  REQUIRE(h.setOptionValue(kMipLpSolverString, kIpxString) == HighsStatus::kOk);
  REQUIRE(h.setOptionValue(kMipLpSolverString, kPdlpString) ==
          HighsStatus::kError);

  REQUIRE(h.setOptionValue(kMipIpmSolverString, "fred") == HighsStatus::kError);
  REQUIRE(h.setOptionValue(kMipIpmSolverString, kHighsChooseString) ==
          HighsStatus::kOk);
  REQUIRE(h.setOptionValue(kMipIpmSolverString, kSimplexString) ==
          HighsStatus::kError);
  REQUIRE(h.setOptionValue(kMipIpmSolverString, kIpmString) ==
          HighsStatus::kOk);

#ifdef HIPO
  REQUIRE(h.setOptionValue(kMipIpmSolverString, kHipoString) ==
          HighsStatus::kOk);
#else
  REQUIRE(h.setOptionValue(kMipIpmSolverString, kHipoString) ==
          HighsStatus::kError);
#endif

  REQUIRE(h.setOptionValue(kMipIpmSolverString, kIpxString) ==
          HighsStatus::kOk);
  REQUIRE(h.setOptionValue(kMipIpmSolverString, kPdlpString) ==
          HighsStatus::kError);
}

TEST_CASE("mip-lp-solver", "[highs_test_mip_solver]") {
  std::string model_file =
      std::string(HIGHS_DIR) + "/check/instances/flugpl.mps";
  Highs h;
  h.setOptionValue("output_flag", dev_run);
  const bool just_hipo_test = false;
  if (!just_hipo_test) {
    REQUIRE(h.readModel(model_file) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);

    REQUIRE(h.readModel(model_file) == HighsStatus::kOk);
    REQUIRE(h.setOptionValue(kMipLpSolverString, kIpxString) ==
            HighsStatus::kOk);
    REQUIRE(h.setOptionValue(kMipIpmSolverString, kIpxString) ==
            HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  }
#ifdef HIPO
  REQUIRE(h.readModel(model_file) == HighsStatus::kOk);
  REQUIRE(h.setOptionValue(kMipLpSolverString, kHipoString) ==
          HighsStatus::kOk);
  REQUIRE(h.setOptionValue(kMipIpmSolverString, kHipoString) ==
          HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
#endif
}

/*
TEST_CASE("mip-sub-solver-time", "[highs_test_mip_solver]") {
  const std::string model = "flugpl";  //"rgn"; //
  std::string model_file =
      std::string(HIGHS_DIR) + "/check/instances/" + model + ".mps";
  Highs h;
  h.setOptionValue("output_flag", dev_run);
  h.setOptionValue("highs_analysis_level", kHighsAnalysisLevelMipTime);
  REQUIRE(h.readModel(model_file) == HighsStatus::kOk);

  REQUIRE(h.run() == HighsStatus::kOk);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
}
*/

TEST_CASE("get-fixed-lp", "[highs_test_mip_solver]") {
  std::string model = "avgas";
  std::string model_file =
      std::string(HIGHS_DIR) + "/check/instances/" + model + ".mps";
  HighsLp fixed_lp;
  Highs h;
  h.setOptionValue("output_flag", dev_run);
  REQUIRE(h.readModel(model_file) == HighsStatus::kOk);
  REQUIRE(h.getFixedLp(fixed_lp) == HighsStatus::kError);

  model = "flugpl";
  model_file = std::string(HIGHS_DIR) + "/check/instances/" + model + ".mps";
  REQUIRE(h.readModel(model_file) == HighsStatus::kOk);
  REQUIRE(h.getFixedLp(fixed_lp) == HighsStatus::kError);

  REQUIRE(h.run() == HighsStatus::kOk);
  double mip_optimal_objective = h.getInfo().objective_function_value;
  HighsSolution solution = h.getSolution();

  // Transform the incumbent MIP into the fixed LP
  HighsLp mip = h.getLp();
  std::vector<HighsInt> col_set;
  std::vector<double> fixed_value;
  for (HighsInt iCol = 0; iCol < mip.num_col_; iCol++) {
    if (mip.integrality_[iCol] == HighsVarType::kInteger) {
      col_set.push_back(iCol);
      fixed_value.push_back(solution.col_value[iCol]);
    }
  }
  h.clearIntegrality();
  HighsInt num_set_entries = col_set.size();
  h.changeColsBounds(num_set_entries, col_set.data(), fixed_value.data(),
                     fixed_value.data());
  h.setOptionValue("presolve", kHighsOffString);
  REQUIRE(h.run() == HighsStatus::kOk);

  REQUIRE(std::abs(h.getInfo().objective_function_value -
                   mip_optimal_objective) < double_equal_tolerance);
  // In calling changeColsBounds, the incumbent solution was always
  // cleared, so there was no information from which to construct an
  // advanced basis. Hence simplex starts from a logical basis and
  // requires a positive number of iterations (#2556)
  //
  // Before code to retain solution if changing the bounds and
  // solution remains feasible
  //
  //  REQUIRE(h.getInfo().simplex_iteration_count > 0);
  REQUIRE(h.getInfo().simplex_iteration_count == 0);

  // Now, passing the MIP solution, there is information from which to
  // construct an advanced basis. In the case of flugpl, this is
  // optimal, so no simplex iterations are required
  h.clearSolver();
  h.setSolution(solution);
  REQUIRE(h.run() == HighsStatus::kOk);

  REQUIRE(std::abs(h.getInfo().objective_function_value -
                   mip_optimal_objective) < double_equal_tolerance);
  REQUIRE(h.getInfo().simplex_iteration_count == 0);

  // Now re-load the MIP, re-solve, and get the fixed LP
  REQUIRE(h.passModel(mip) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);

  // REQUIRE(h.getInfo().objective_function_value == mip_optimal_objective);
  REQUIRE(objectiveOk(mip_optimal_objective,
                      h.getInfo().objective_function_value, dev_run));

  REQUIRE(h.getFixedLp(fixed_lp) == HighsStatus::kOk);

  REQUIRE(h.passModel(fixed_lp) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);

  REQUIRE(std::abs(h.getInfo().objective_function_value -
                   mip_optimal_objective) < double_equal_tolerance);

  // Now run from saved solution (without presolve)
  h.clearSolver();
  h.setSolution(solution);
  REQUIRE(h.run() == HighsStatus::kOk);

  REQUIRE(std::abs(h.getInfo().objective_function_value -
                   mip_optimal_objective) < double_equal_tolerance);
  REQUIRE(h.getInfo().simplex_iteration_count == 0);

  REQUIRE(h.readModel(model_file) == HighsStatus::kOk);
  // Perturb one of the integer variables for code coverage of
  // warning: makes fixed LP of flugpl infeasible
  std::vector<HighsVarType> integrality = h.getLp().integrality_;
  for (HighsInt iCol = 0; iCol < fixed_lp.num_col_; iCol++) {
    if (integrality[iCol] != HighsVarType::kContinuous) {
      solution.col_value[iCol] -= 0.01;
      break;
    }
  }

  REQUIRE(h.run() == HighsStatus::kOk);
  h.setSolution(solution);

  REQUIRE(h.getFixedLp(fixed_lp) == HighsStatus::kWarning);

  REQUIRE(h.passModel(fixed_lp) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);

  REQUIRE(h.getModelStatus() == HighsModelStatus::kInfeasible);

  h.resetGlobalScheduler(true);
}

TEST_CASE("get-presolved-mip", "[highs_test_mip_solver]") {
  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 3;
  lp.col_cost_ = {1, 1, 1};
  lp.col_lower_ = {0, -kHighsInf, -kHighsInf};
  lp.col_upper_ = {kHighsInf, kHighsInf, kHighsInf};
  lp.integrality_ = {HighsVarType::kContinuous, HighsVarType::kInteger,
                     HighsVarType::kInteger};
  lp.row_lower_ = {2, 6, 8};
  lp.row_upper_ = {2, kHighsInf, kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 3, 6, 9};
  lp.a_matrix_.index_ = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  lp.a_matrix_.value_ = {1, 1, 1, 1, -1, 2, 1, 3, -1};
  Highs h;
  h.setOptionValue("output_flag", dev_run);
  // Code coverage of highsVarTypeToString for all cases
  HighsLogOptions log_options = h.getOptions().log_options;
  for (HighsInt iVarType = -1;
       iVarType < HighsInt(HighsVarType::kImplicitInteger) + 2; iVarType++)
    highsLogUser(log_options, HighsLogType::kInfo, "Variable type %2d is %s\n",
                 int(iVarType), highsVarTypeToString(iVarType).c_str());
  h.passModel(lp);
  h.presolve();
  // Presolved MIP has an implied integer, so this tests passing such
  HighsLp presolved_lp = h.getPresolvedModel().lp_;
  h.run();
  const double lp_objective_value = h.getObjectiveValue();
  h.passModel(presolved_lp);
  h.run();
  const double presolved_lp_objective_value = h.getObjectiveValue();
  REQUIRE(presolved_lp_objective_value == lp_objective_value);
  h.resetGlobalScheduler(true);
}

TEST_CASE("get-fixed-lp-semi", "[highs_test_mip_solver]") {
  HighsLp lp;
  lp.num_col_ = 4;
  lp.num_row_ = 2;
  lp.col_cost_ = {1, 3, 1, 2};
  lp.col_lower_ = {0, 0, 1, 1};
  lp.col_upper_ = {1, 1, 3, 5};
  lp.integrality_ = {HighsVarType::kContinuous, HighsVarType::kInteger,
                     HighsVarType::kSemiContinuous, HighsVarType::kSemiInteger};
  lp.row_lower_ = {4, 10};
  lp.row_upper_ = {kHighsInf, kHighsInf};
  lp.a_matrix_.start_ = {0, 2, 4, 6, 8};
  lp.a_matrix_.index_ = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  lp.a_matrix_.value_ = {1, 1, 1, 2, 1, 3, 1, 4, 5, 1};
  Highs h;
  h.setOptionValue("output_flag", dev_run);
  h.setOptionValue("presolve", kHighsOffString);
  // Code coverage of highsVarTypeToString for four main types
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++)
    highsLogUser(h.getOptions().log_options, HighsLogType::kInfo,
                 "Column %d is of type %s\n", int(iCol),
                 highsVarTypeToString(lp.integrality_[iCol]).c_str());
  h.passModel(lp);
  h.run();
  double mip_optimal_objective = h.getInfo().objective_function_value;
  HighsSolution solution = h.getSolution();
  HighsLp fixed_lp;
  REQUIRE(h.getFixedLp(fixed_lp) == HighsStatus::kOk);

  REQUIRE(h.passModel(fixed_lp) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);

  REQUIRE(h.getInfo().objective_function_value == mip_optimal_objective);
  h.resetGlobalScheduler(true);
}

TEST_CASE("row-fixed-lp", "[highs_test_mip_solver]") {
  std::string model = "flugpl";
  std::string model_file =
      std::string(HIGHS_DIR) + "/check/instances/" + model + ".mps";
  Highs h;
  h.setOptionValue("output_flag", dev_run);
  REQUIRE(h.readModel(model_file) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);
  double mip_optimal_objective = h.getInfo().objective_function_value;
  HighsSolution solution = h.getSolution();

  HighsLp lp = h.getLp();
  h.clearIntegrality();
  h.changeRowsBounds(0, lp.num_row_ - 1, solution.row_value.data(),
                     solution.row_value.data());
  h.setOptionValue("presolve", kHighsOffString);
  REQUIRE(h.run() == HighsStatus::kOk);
  REQUIRE(h.getInfo().objective_function_value <=
          mip_optimal_objective + double_equal_tolerance);

  h.resetGlobalScheduler(true);
}

TEST_CASE("issue-2585", "[highs_test_mip_solver]") {
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/issue-2585.lp";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.readModel(filename);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = -175.91;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("issue-2173", "[highs_test_mip_solver]") {
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/issue-2173.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.readModel(filename);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = -26770.8075489;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("parallel-mip-determinism", "[highs_test_mip_solver]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/bell5.mps";
  HighsInt num_runs = 6;
  std::vector<HighsInt> lp_iters(num_runs);
  for (HighsInt i = 0; i < num_runs; i++) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("threads", 2);
    highs.setOptionValue("parallel", kHighsOnString);
    if (i % 2 == 0) highs.setOptionValue("mip_search_simulate_concurrency", 1);
    highs.readModel(filename);
    const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
    const double optimal_objective = 8966406.491519;
    solve(highs, kHighsOffString, require_model_status, optimal_objective);
    lp_iters[i] = highs.getInfo().simplex_iteration_count;
    if (i > 0) {
      REQUIRE(lp_iters[i] == lp_iters[0]);
    }
  }
}

TEST_CASE("issue-2957", "[highs_test_mip_solver]") {
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {1, 2};
  lp.col_lower_ = {0, 8};
  lp.col_upper_ = {20, 20};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
  lp.row_lower_ = {20.1};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.passModel(lp);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = 28.2;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("issue-2975", "[highs_test_mip_solver]") {
  //   min  2*b + 99999*y
  //   s.t. a + b = 10
  //        a - 100*y <= 0
  //        a, b >= 0;  y binary
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  HighsInt a = 0;
  HighsInt b = 1;
  HighsInt y = 2;
  HighsLp lp;
  lp.num_col_ = 3;
  lp.num_row_ = 2;
  lp.col_lower_ = {0, 0, 0};
  lp.col_upper_ = {kHighsInf, kHighsInf, 1};
  lp.col_cost_ = {0, 2, 99999};
  lp.integrality_ = {HighsVarType::kContinuous, HighsVarType::kContinuous,
                     HighsVarType::kInteger};
  lp.row_lower_ = {10, -kHighsInf};
  lp.row_upper_ = {10, 0};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 2, 4};
  lp.a_matrix_.index_ = {a, b, a, y};
  lp.a_matrix_.value_ = {1, 1, 1, -100};
  highs.passModel(lp);
  highs.run();
  REQUIRE(highs.getInfo().objective_function_value == 20);
  REQUIRE(highs.getSolution().col_value[y] == 0.0);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("issue-3118", "[highs_test_mip_solver]") {
  const double M = 1e10;
  //   min    x +   y
  //   s.t.   x + M*y = 1
  //        M*x +   y = 1
  //          x, y binary
  //   Initial "solution" x = y = 1/M
  //
  // This problem is found infeasible in presolve, but the point x = y
  // = 1/M is integer feasible to within the tolerances, so is
  // acccepted as an optimal solution to the problem
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);

  HighsInt x = 0;
  HighsInt y = 1;
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 2;
  lp.col_lower_ = {0., 0.};
  lp.col_upper_ = {1., 1.};
  lp.col_cost_ = {1., 1.};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  lp.row_lower_ = {1., 1.};
  lp.row_upper_ = {1., 1.};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 2, 4};
  lp.a_matrix_.index_ = {x, y, x, y};
  lp.a_matrix_.value_ = {1., M, M, 1};
  highs.passModel(lp);

  std::vector<double> solution_values(lp.num_col_, 1 / M);
  highs.setSolution(2, nullptr, solution_values.data());

  highs.run();
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(std::abs(2 / M - highs.getInfo().objective_function_value) < 1e-9);
  if (dev_run) highs.writeSolution("", 1);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("issue-3118a", "[highs_test_mip_solver]") {
  const double M = 1e10;
  //   max f = x
  //   s.t.    x - M*y  = -1
  //           x       <=  b
  //           x, y integer in [0, 2]x[0, 1]
  //
  // The MIP is not feasible over the integers for any b
  //
  // With initial "solution" x = 0; y = 1/M
  //
  // For b = 0: (0, 1/M) with f = 0 is the only feasible solution of
  // the relaxation, and it's also feasible for the MIP, so claiming
  // it is optimal when presolve identifies infeasibility is clearly
  // justified
  //
  // For b = 1: (1, 2/M) with f = 1 is the optimal solution of the
  // relaxation, and it's also feasible for the MIP. However, claiming
  // that the initial "solution" (0, 1/M) with f = 0 is optimal when
  // presolve identifies infeasibility is still justified, because
  // it's a point that is feasible for a MIP that is deemed infeasible
  // by presolve
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);

  HighsInt x = 0;
  HighsInt y = 1;
  HighsLp lp;
  lp.sense_ = ObjSense::kMaximize;
  lp.num_col_ = 2;
  lp.num_row_ = 2;
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {2, 1};
  lp.col_cost_ = {1, 0};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  lp.row_lower_ = {-1, -kHighsInf};
  lp.row_upper_ = {-1, 0};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 2, 3};
  lp.a_matrix_.index_ = {x, y, x};
  lp.a_matrix_.value_ = {1., -M, 1};

  for (HighsInt k = 0; k < 2; k++) {
    HighsInt b = k == 0 ? 0 : 1;
    lp.row_upper_[1] = b;

    highs.passModel(lp);

    // Solve as MIP
    if (dev_run)
      printf("================\nCase b = %d (MIP)\n================\n", int(b));
    highs.setOptionValue("solve_relaxation", false);
    std::vector<double> solution_values = {0, 1 / M};
    highs.setSolution(2, nullptr, solution_values.data());

    highs.run();
    if (dev_run) highs.writeSolution("", 1);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);

    // Solve as LP
    if (dev_run)
      printf("===============\nCase b = %d (LP)\n===============\n", int(b));
    highs.setOptionValue("solve_relaxation", true);
    highs.clearSolver();
    highs.run();
    highs.writeSolution("", 1);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double lp_objective_value = highs.getInfo().objective_function_value;

    solution_values = highs.getSolution().col_value;
    highs.setSolution(2, nullptr, solution_values.data());

    bool valid, integral, feasible;
    REQUIRE(highs.assessPrimalSolution(valid, integral, feasible) ==
            HighsStatus::kOk);
  }

  highs.resetGlobalScheduler(true);
}

// Independent-component decomposition: block-diagonal binary knapsacks
// with a known closed-form optimum. Each block is far below the exact
// subsolve caps, so decomposition must collapse the model; the same
// optimum must come out of the fallback path (mip_decomposition=false).
static double decompBlockOptimum(HighsInt blockCols, HighsInt cap) {
  double optimum = 0.0;
  for (HighsInt i = 0; i < cap; ++i) optimum -= (blockCols - i);
  return optimum;
}

static HighsLp decompBlockKnapsack(HighsInt numBlocks, HighsInt blockCols,
                                   HighsInt cap) {
  HighsLp lp;
  lp.num_col_ = numBlocks * blockCols;
  lp.num_row_ = numBlocks;
  lp.sense_ = ObjSense::kMinimize;
  lp.col_cost_.resize(lp.num_col_);
  lp.col_lower_.assign(lp.num_col_, 0.0);
  lp.col_upper_.assign(lp.num_col_, 1.0);
  lp.integrality_.assign(lp.num_col_, HighsVarType::kInteger);
  lp.row_lower_.assign(lp.num_row_, -kHighsInf);
  lp.row_upper_.assign(lp.num_row_, double(cap));
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.resize(lp.num_col_ + 1);
  for (HighsInt c = 0; c < lp.num_col_; ++c) {
    lp.col_cost_[c] = -double(c % blockCols + 1);
    lp.a_matrix_.start_[c + 1] = c + 1;
    lp.a_matrix_.index_.push_back(c / blockCols);
    lp.a_matrix_.value_.push_back(1.0);
  }
  return lp;
}

TEST_CASE("MIP-decomposition-blocks", "[highs_test_mip_solver]") {
  // 6 blocks x 20 binary cols = 120 cols: above the toy-model skip, each
  // block far below the exact-subsolve caps.
  HighsLp lp = decompBlockKnapsack(6, 20, 10);
  const double optimal_objective = 6 * decompBlockOptimum(20, 10);
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.passModel(lp);
  solve(highs, kHighsOnString, HighsModelStatus::kOptimal, optimal_objective);
  // Fallback path must give the identical optimum.
  REQUIRE(highs.setOptionValue("mip_decomposition", false) ==
          HighsStatus::kOk);
  highs.clearSolver();
  solve(highs, kHighsOnString, HighsModelStatus::kOptimal, optimal_objective);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("MIP-decomposition-infeasible", "[highs_test_mip_solver]") {
  // One provably infeasible 2-variable block padded with 100 isolated
  // binaries (102 columns, above the toy-model skip): an infeasible
  // exact subsolve must propagate to global infeasibility, and the
  // fallback path must agree.
  HighsLp lp;
  lp.num_col_ = 102;
  lp.num_row_ = 2;
  lp.sense_ = ObjSense::kMinimize;
  lp.col_cost_.assign(lp.num_col_, 1.0);
  lp.col_lower_.assign(lp.num_col_, 0.0);
  lp.col_upper_.assign(lp.num_col_, 1.0);
  lp.integrality_.assign(lp.num_col_, HighsVarType::kInteger);
  lp.row_lower_ = {2.0, -kHighsInf};
  lp.row_upper_ = {kHighsInf, 1.0};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.resize(lp.num_col_ + 1);
  lp.a_matrix_.start_[0] = 0;
  lp.a_matrix_.start_[1] = 2;
  lp.a_matrix_.start_[2] = 4;
  for (HighsInt c = 2; c < lp.num_col_; ++c) lp.a_matrix_.start_[c + 1] = 4;
  lp.a_matrix_.index_ = {0, 1, 0, 1};
  lp.a_matrix_.value_ = {1.0, 1.0, 1.0, 1.0};
  for (bool decomposition : {true, false}) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_decomposition", decomposition);
    highs.passModel(lp);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kInfeasible);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-decomposition-logging", "[highs_test_mip_solver]") {
  // Single large block (nothing eligible) with the log-only analyser on:
  // exercises the weak-coupling detector without changing the result.
  HighsLp lp = decompBlockKnapsack(1, 120, 60);
  const double optimal_objective = decompBlockOptimum(120, 60);
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.setOptionValue("mip_decomposition_logging", true);
  highs.passModel(lp);
  solve(highs, kHighsOnString, HighsModelStatus::kOptimal, optimal_objective);

  highs.resetGlobalScheduler(true);
}

// Classical Benders decomposition: arrowhead models with continuous
// blocks and binary coupling columns. The optimum is not known in closed
// form, so each test requires optimality plus on/off agreement between
// the Benders path and the fallback path (mip_benders=false).
static HighsLp bendersArrowhead(bool maximize) {
  // 3 blocks x 40 continuous cols + y1, y2 binary couplers.
  const HighsInt nb = 40;
  HighsLp lp;
  lp.num_col_ = 3 * nb + 2;
  lp.num_row_ = 3 * 4 + 3;
  lp.sense_ = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  const double s = maximize ? -1.0 : 1.0;
  lp.col_cost_.resize(lp.num_col_);
  lp.col_lower_.assign(lp.num_col_, 0.0);
  lp.col_upper_.assign(lp.num_col_, 10.0);
  lp.integrality_.assign(lp.num_col_, HighsVarType::kContinuous);
  const HighsInt y1 = 3 * nb;
  const HighsInt y2 = 3 * nb + 1;
  lp.col_upper_[y1] = 1.0;
  lp.col_upper_[y2] = 1.0;
  lp.integrality_[y1] = HighsVarType::kInteger;
  lp.integrality_[y2] = HighsVarType::kInteger;
  lp.col_cost_[y1] = s * 500.0;
  lp.col_cost_[y2] = s * 700.0;
  lp.row_lower_.resize(lp.num_row_);
  lp.row_upper_.resize(lp.num_row_);
  for (HighsInt b = 0; b < 3; ++b) {
    lp.row_lower_[4 * b + 0] = -kHighsInf;  // knapsack
    lp.row_upper_[4 * b + 0] = 300.0;
    lp.row_lower_[4 * b + 1] = 400.0;  // covering
    lp.row_upper_[4 * b + 1] = kHighsInf;
    lp.row_lower_[4 * b + 2] = 0.0;  // alternating equality
    lp.row_upper_[4 * b + 2] = 0.0;
    lp.row_lower_[4 * b + 3] = 10.0;  // ranged difference rows
    lp.row_upper_[4 * b + 3] = 50.0;
  }
  for (HighsInt t = 0; t < 3; ++t) {
    lp.row_lower_[12 + t] = -kHighsInf;  // coupling rows
    lp.row_upper_[12 + t] = 0.0;
  }
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.resize(lp.num_col_ + 1);
  for (HighsInt b = 0; b < 3; ++b) {
    for (HighsInt i = 0; i < nb; ++i) {
      HighsInt c = b * nb + i;
      lp.col_cost_[c] = s * double((i % 7) - 3);
      lp.a_matrix_.start_[c + 1] = (HighsInt)lp.a_matrix_.index_.size() + 5;
      lp.a_matrix_.index_.push_back(4 * b + 0);
      lp.a_matrix_.value_.push_back(1.0);
      lp.a_matrix_.index_.push_back(4 * b + 1);
      lp.a_matrix_.value_.push_back(double((i % 5) + 1));
      lp.a_matrix_.index_.push_back(4 * b + 2);
      lp.a_matrix_.value_.push_back(i % 2 ? -1.0 : 1.0);
      lp.a_matrix_.index_.push_back(4 * b + 3);
      lp.a_matrix_.value_.push_back(i < nb / 2 ? 1.0 : -1.0);
      if (b == 0 && i < 3) {
        lp.a_matrix_.index_.push_back(12);
        lp.a_matrix_.value_.push_back(1.0);
      } else if (b == 1 && i >= 3 && i < 6) {
        lp.a_matrix_.index_.push_back(13);
        lp.a_matrix_.value_.push_back(1.0);
      } else if (b == 2 && i == 0) {
        lp.a_matrix_.index_.push_back(14);
        lp.a_matrix_.value_.push_back(1.0);
      } else {
        lp.a_matrix_.start_[c + 1]--;
      }
    }
  }
  lp.a_matrix_.start_[y1 + 1] =
      (HighsInt)lp.a_matrix_.index_.size() + 2;
  lp.a_matrix_.index_.push_back(12);
  lp.a_matrix_.value_.push_back(-30.0);
  lp.a_matrix_.index_.push_back(14);
  lp.a_matrix_.value_.push_back(-10.0);
  lp.a_matrix_.start_[y2 + 1] =
      (HighsInt)lp.a_matrix_.index_.size() + 2;
  lp.a_matrix_.index_.push_back(13);
  lp.a_matrix_.value_.push_back(-30.0);
  lp.a_matrix_.index_.push_back(14);
  lp.a_matrix_.value_.push_back(-10.0);
  return lp;
}

static double runBendersOnOff(HighsLp lp, bool decomposition_logging = false,
                              bool small_blocks = false) {
  // Returns the Benders-path objective after requiring optimality and
  // exact agreement with the fallback path.
  double objective_on = kHighsInf;
  for (bool benders : {true, false}) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", benders);
    highs.setOptionValue("mip_decomposition_logging", decomposition_logging);
    if (small_blocks) {
      highs.setOptionValue("mip_decomposition_max_comp_cols", 5);
      highs.setOptionValue("mip_decomposition_max_comp_rows", 5);
      highs.setOptionValue("mip_benders_min_block_cols", 4);
    }
    highs.passModel(lp);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (benders)
      objective_on = objective;
    else
      REQUIRE(objective == objective_on);
    highs.resetGlobalScheduler(true);
  }
  return objective_on;
}

TEST_CASE("MIP-benders-arrowhead", "[highs_test_mip_solver]") {
  runBendersOnOff(bendersArrowhead(false));
}

TEST_CASE("MIP-benders-max", "[highs_test_mip_solver]") {
  runBendersOnOff(bendersArrowhead(true));
}

TEST_CASE("MIP-benders-farkas", "[highs_test_mip_solver]") {
  // y1=1 renders the A/B blocks LP-infeasible (needs fractional Farkas
  // multipliers, invisible to presolve/probing), so the master must learn
  // feasibility cuts. Small-block options isolate the ray path.
  HighsLp lp;
  lp.num_col_ = 30 + 30 + 60 + 1;
  lp.num_row_ = 3 + 3 + 4;
  lp.sense_ = ObjSense::kMinimize;
  lp.col_cost_.assign(lp.num_col_, 0.0);
  lp.col_lower_.assign(lp.num_col_, 0.0);
  lp.col_upper_.assign(lp.num_col_, 10.0);
  lp.integrality_.assign(lp.num_col_, HighsVarType::kContinuous);
  const HighsInt y1 = lp.num_col_ - 1;
  lp.col_upper_[y1] = 1.0;
  lp.integrality_[y1] = HighsVarType::kInteger;
  lp.col_cost_[y1] = -500.0;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.resize(lp.num_col_ + 1);
  auto addEntry = [&](HighsInt c, HighsInt r, double v) {
    lp.a_matrix_.index_.push_back(r);
    lp.a_matrix_.value_.push_back(v);
    for (HighsInt cc = c + 1; cc <= lp.num_col_; ++cc)
      lp.a_matrix_.start_[cc]++;
  };
  // A/B blocks (cols 0..29, 30..59): r1/r2/c2 rows (0..2, 3..5).
  for (HighsInt t = 0; t < 2; ++t) {
    for (HighsInt i = 0; i < 30; ++i) {
      HighsInt c = t * 30 + i;
      lp.col_cost_[c] = double((i % 3) - 1);
      const bool even = (i % 2 == 0);
      addEntry(c, 3 * t + 0, even ? 1.0 : -1.0);
      addEntry(c, 3 * t + 1, even ? -3.0 : 2.0);
      addEntry(c, 3 * t + 2, 1.0);
    }
  }
  addEntry(y1, 1, -450.0);
  addEntry(y1, 2, 50.0);
  addEntry(y1, 4, -450.0);
  addEntry(y1, 5, 50.0);
  // Padding block (cols 60..119): knapsack/covering/equality/diffcap.
  for (HighsInt i = 0; i < 60; ++i) {
    HighsInt c = 60 + i;
    lp.col_cost_[c] = double((i % 7) - 3);
    addEntry(c, 6, 1.0);
    addEntry(c, 7, double((i % 5) + 1));
    addEntry(c, 8, i % 2 ? -1.0 : 1.0);
    addEntry(c, 9, i < 30 ? 1.0 : -1.0);
  }
  lp.row_lower_ = {40.0, -480.0, -kHighsInf, 40.0, -480.0,
                   -kHighsInf, -kHighsInf, 400.0, 0.0, -kHighsInf};
  lp.row_upper_ = {kHighsInf, kHighsInf, 350.0, kHighsInf, kHighsInf,
                   350.0, 300.0, kHighsInf, 0.0, 50.0};
  runBendersOnOff(lp, false, true);
}

static HighsLp bendersFeasAuxModel() {
  // Compact feasibility-cut model (same recipe as MIP-benders-farkas):
  // y1=1 renders the A/B blocks LP-infeasible, so the master must learn
  // feasibility cuts via rays or via the auxiliary fallback.
  HighsLp lp;
  lp.num_col_ = 25 + 25 + 50 + 1;
  lp.num_row_ = 3 + 3 + 4;
  lp.sense_ = ObjSense::kMinimize;
  lp.col_cost_.assign(lp.num_col_, 0.0);
  lp.col_lower_.assign(lp.num_col_, 0.0);
  lp.col_upper_.assign(lp.num_col_, 10.0);
  lp.integrality_.assign(lp.num_col_, HighsVarType::kContinuous);
  const HighsInt y1 = lp.num_col_ - 1;
  lp.col_upper_[y1] = 1.0;
  lp.integrality_[y1] = HighsVarType::kInteger;
  lp.col_cost_[y1] = -500.0;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.resize(lp.num_col_ + 1);
  auto addEntry = [&](HighsInt c, HighsInt r, double v) {
    lp.a_matrix_.index_.push_back(r);
    lp.a_matrix_.value_.push_back(v);
    for (HighsInt cc = c + 1; cc <= lp.num_col_; ++cc)
      lp.a_matrix_.start_[cc]++;
  };
  for (HighsInt t = 0; t < 2; ++t) {
    for (HighsInt i = 0; i < 25; ++i) {
      HighsInt c = t * 25 + i;
      lp.col_cost_[c] = double((i % 3) - 1);
      const bool even = (i % 2 == 0);
      addEntry(c, 3 * t + 0, even ? 1.0 : -1.0);
      addEntry(c, 3 * t + 1, even ? -3.0 : 2.0);
      addEntry(c, 3 * t + 2, 1.0);
    }
  }
  addEntry(y1, 1, -450.0);
  addEntry(y1, 2, 50.0);
  addEntry(y1, 4, -450.0);
  addEntry(y1, 5, 50.0);
  for (HighsInt i = 0; i < 50; ++i) {
    HighsInt c = 50 + i;
    lp.col_cost_[c] = double((i % 7) - 3);
    addEntry(c, 6, 1.0);
    addEntry(c, 7, double((i % 5) + 1));
    addEntry(c, 8, i % 2 ? -1.0 : 1.0);
    addEntry(c, 9, i < 25 ? 1.0 : -1.0);
  }
  lp.row_lower_ = {40.0, -480.0, -kHighsInf, 40.0, -480.0,
                   -kHighsInf, -kHighsInf, 400.0, 0.0, -kHighsInf};
  lp.row_upper_ = {kHighsInf, kHighsInf, 350.0, kHighsInf, kHighsInf,
                   350.0, 300.0, kHighsInf, 0.0, 50.0};
  return lp;
}

TEST_CASE("MIP-benders-feasaux", "[highs_test_mip_solver]") {
  // Aux-fallback toggle must never change the proven optimum: aux on,
  // aux off (pure ray path), and Benders off must all agree exactly.
  HighsLp lp = bendersFeasAuxModel();
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 3; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", cfg != 2);
    highs.setOptionValue("mip_benders_feas_aux", cfg == 0);
    highs.setOptionValue("mip_decomposition_max_comp_cols", 5);
    highs.setOptionValue("mip_decomposition_max_comp_rows", 5);
    highs.setOptionValue("mip_benders_min_block_cols", 4);
    highs.passModel(lp);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-integer-subproblems", "[highs_test_mip_solver]") {
  // Arrowhead with integer blocks (LP-tight single knapsacks plus a
  // parity block): exercises LP-relaxation cuts, sub-MIP upper bounds
  // and the no-good path machinery with on/off agreement. Presolve
  // rules are switched off by bit mask (bits 6..19 = all allow-off
  // rules) so the arrowhead structure survives to Benders; the test
  // pins optimality and agreement, not presolve behavior.
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-int.lp";
  double objective_on = kHighsInf;
  for (bool benders : {true, false}) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", benders);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (benders)
      objective_on = objective;
    else
      REQUIRE(objective == objective_on);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-lshaped", "[highs_test_mip_solver]") {
  // Integer L-shaped toggle must never change the proven optimum: five
  // binary-coupled MIP blocks with LP-loose equalities (L-shaped cuts
  // fire from iteration 0 and converge) agree exactly with L-shaped on,
  // L-shaped off (LP-relax cuts + sub-MIP bounds only), and Benders off.
  // Presolve rules are switched off by bit mask so the arrowhead
  // structure survives to Benders.
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 3; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", cfg != 2);
    highs.setOptionValue("mip_benders_lshaped", cfg == 0);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-stall", "[highs_test_mip_solver]") {
  // Stall limit must only stop the loop early, never change the proven
  // optimum: with mip_benders_stall_limit = 1 the L-shaped model stops
  // at the first LB stall and falls back, still agreeing exactly with
  // Benders off.
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 2; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", cfg == 0);
    highs.setOptionValue("mip_benders_stall_limit", 1);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-rescue", "[highs_test_mip_solver]") {
  // Fallback-UB rescue: with mip_benders_max_iterations = 1 the loop
  // stops after one iteration holding a feasible composition (UB 216);
  // the verified composition must be injected as incumbent, and all
  // three configurations (rescue on/off, Benders off) agree exactly.
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 3; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", cfg != 2);
    highs.setOptionValue("mip_benders_incumbent", cfg == 0);
    highs.setOptionValue("mip_benders_max_iterations", 1);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-dec", "[highs_test_mip_solver]") {
  // Annotation override: name-based and index-based .dec files drive
  // Benders to the proven optimum; garbage and unreadable files fall
  // back safely. All configurations agree exactly with Benders off.
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  std::string decNames =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.dec";
  std::string decIdx =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped-idx.dec";
  std::string decBad =
      std::string(HIGHS_DIR) + "/check/instances/benders-bad.dec";
  std::vector<std::string> decFiles = {decNames, decIdx, decBad,
                                       "/nonexistent.dec", ""};
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 6; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", cfg != 5);
    if (cfg < 5) highs.setOptionValue("mip_benders_dec_file", decFiles[cfg]);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-maxcuts", "[highs_test_mip_solver]") {
  // Cut cap: with mip_benders_max_cuts = 2 the loop is starved of cuts
  // (priority trimming fires) and falls back, yet the rescued incumbent
  // and the fallback path agree exactly with uncapped and Benders-off.
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 3; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", cfg != 2);
    highs.setOptionValue("mip_benders_max_cuts", cfg == 0 ? 2 : 1000000);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-branchpriority", "[highs_test_mip_solver]") {
  // Branch-on-bridges bonus must never change the proven optimum: with
  // a stored (unfixed) coupling set, bonus on/off and Benders off agree
  // exactly. (Default 0.0 is bit-identical by construction.)
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 3; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", cfg != 2);
    highs.setOptionValue("mip_benders_max_iterations", 1);
    highs.setOptionValue("mip_benders_branch_priority", cfg == 0 ? 1e6 : 0.0);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-dualbound", "[highs_test_mip_solver]") {
  // Injected master lower bounds must never change the proven optimum:
  // dual-bound on/off and Benders off agree exactly (minimize).
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 3; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders", cfg != 2);
    highs.setOptionValue("mip_benders_dual_bound", cfg == 0);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-benders-dualbound-max", "[highs_test_mip_solver]") {
  // Maximize sense: the bound sits on the upper side and must be
  // ignored safely (on/off agreement at the mirrored optimum).
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 2; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_benders_dual_bound", cfg == 0);
    highs.passModel(bendersArrowhead(true));
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-heuristic-polish", "[highs_test_mip_solver]") {
  // LP-free polish must never change the proven optimum: a model whose
  // rounded incumbents admit single-shift improvement (min 2x + y over
  // x + y >= 1: (1,1) polishes to (0,1)) agrees exactly with polish
  // on/off. (Both paths prove 1.)
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.sense_ = ObjSense::kMinimize;
  lp.col_cost_ = {2.0, 1.0};
  lp.col_lower_ = {0.0, 0.0};
  lp.col_upper_ = {1.0, 1.0};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  lp.row_lower_ = {1.0};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1.0, 1.0};
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 2; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_heuristic_polish", cfg == 0);
    highs.passModel(lp);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(objective == 1.0);
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-heuristic-hamming", "[highs_test_mip_solver]") {
  // Hamming-ball search must never change the proven optimum: on/off
  // agree exactly (sub-MIP neighbourhood around the incumbent).
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 2; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_heuristic_run_hamming", cfg == 0);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-heuristic-proximity", "[highs_test_mip_solver]") {
  // Proximity search must never change the proven optimum: on/off
  // agree exactly (distance-to-incumbent sub-MIP under strict cutoff).
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/benders-lshaped.lp";
  double reference = kHighsInf;
  for (HighsInt cfg = 0; cfg != 2; ++cfg) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_heuristic_run_proximity", cfg == 0);
    highs.setOptionValue("presolve_rule_off", (HighsInt)1048512);
    highs.readModel(filename);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (cfg == 0)
      reference = objective;
    else
      REQUIRE(objective == reference);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-decomposition-zero-objective", "[highs_test_mip_solver]") {
  // Zero objective: every feasible point is optimal; decomposition must
  // still agree with the fallback path (objective 0).
  HighsLp lp = decompBlockKnapsack(6, 20, 10);
  lp.col_cost_.assign(lp.num_col_, 0.0);
  for (bool decomposition : {true, false}) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_decomposition", decomposition);
    highs.passModel(lp);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(highs.getInfo().objective_function_value == 0.0);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-decomposition-unbounded", "[highs_test_mip_solver]") {
  // One unbounded component (free continuous column with cost pushing
  // to infinity, kept in a row so it is not an isolated column) padded
  // with 100 isolated binaries: the exact subsolve must skip it and the
  // parent MIP must report unbounded-or-infeasible (plain HiGHS MIP
  // semantics without allow_unbounded_or_infeasible) on both paths.
  HighsLp lp;
  lp.num_col_ = 101;
  lp.num_row_ = 1;
  lp.sense_ = ObjSense::kMinimize;
  lp.col_cost_.assign(lp.num_col_, 1.0);
  lp.col_cost_[0] = -1.0;
  lp.col_lower_.assign(lp.num_col_, 0.0);
  lp.col_upper_.assign(lp.num_col_, 1.0);
  lp.col_lower_[0] = -kHighsInf;
  lp.col_upper_[0] = kHighsInf;
  lp.integrality_.assign(lp.num_col_, HighsVarType::kInteger);
  lp.integrality_[0] = HighsVarType::kContinuous;
  lp.row_lower_ = {0.0};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.resize(lp.num_col_ + 1);
  lp.a_matrix_.start_[0] = 0;
  lp.a_matrix_.start_[1] = 1;
  for (HighsInt c = 1; c < lp.num_col_; ++c) lp.a_matrix_.start_[c + 1] = 1;
  lp.a_matrix_.index_ = {0};
  lp.a_matrix_.value_ = {1.0};
  for (bool decomposition : {true, false}) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_decomposition", decomposition);
    highs.passModel(lp);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() ==
            HighsModelStatus::kUnboundedOrInfeasible);
    highs.resetGlobalScheduler(true);
  }
}

// Lagrangian decomposition: two continuous LP blocks (above the exact
// subsolve caps so components leaves them) joined by one coupling row,
// plus a tiny integer side block to force the MIP path. Requires
// optimality plus on/off agreement; the Lagrangian loop itself is
// exercised (separator + subgradient + optional injection) while the
// parent MIP always proves.
static HighsLp lagrangianTwoBlocks(double couplingCap) {
  const HighsInt nb = 80;
  HighsLp lp;
  lp.num_col_ = 2 * nb + 10;
  lp.num_row_ = 2 * 3 + 1 + 1;
  lp.sense_ = ObjSense::kMinimize;
  lp.col_cost_.resize(lp.num_col_);
  lp.col_lower_.assign(lp.num_col_, 0.0);
  lp.col_upper_.assign(lp.num_col_, 10.0);
  lp.integrality_.assign(lp.num_col_, HighsVarType::kContinuous);
  lp.row_lower_.resize(lp.num_row_);
  lp.row_upper_.resize(lp.num_row_);
  for (HighsInt b = 0; b < 2; ++b) {
    lp.row_lower_[3 * b + 0] = -kHighsInf;
    lp.row_upper_[3 * b + 0] = 350.0;
    lp.row_lower_[3 * b + 1] = 460.0;
    lp.row_upper_[3 * b + 1] = kHighsInf;
    lp.row_lower_[3 * b + 2] = 0.0;
    lp.row_upper_[3 * b + 2] = 0.0;
  }
  lp.row_lower_[6] = -kHighsInf;  // coupling row over 5+5 vars
  lp.row_upper_[6] = couplingCap;
  lp.row_lower_[7] = -kHighsInf;  // integer side block knapsack
  lp.row_upper_[7] = 5.0;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.resize(lp.num_col_ + 1);
  for (HighsInt b = 0; b < 2; ++b) {
    for (HighsInt i = 0; i < nb; ++i) {
      HighsInt c = b * nb + i;
      lp.col_cost_[c] = double((i % 7) - 3);
      lp.a_matrix_.start_[c + 1] = (HighsInt)lp.a_matrix_.index_.size() + 3;
      lp.a_matrix_.index_.push_back(3 * b + 0);
      lp.a_matrix_.value_.push_back(1.0);
      lp.a_matrix_.index_.push_back(3 * b + 1);
      lp.a_matrix_.value_.push_back(double((i % 5) + 1));
      lp.a_matrix_.index_.push_back(3 * b + 2);
      lp.a_matrix_.value_.push_back(i < nb / 2 ? 1.0 : -1.0);
      if (i < 5) {
        lp.a_matrix_.index_.push_back(6);
        lp.a_matrix_.value_.push_back(1.0);
        lp.a_matrix_.start_[c + 1]++;
      }
    }
  }
  for (HighsInt i = 0; i < 10; ++i) {
    HighsInt c = 2 * nb + i;
    lp.col_cost_[c] = -double(i + 1);
    lp.col_upper_[c] = 1.0;
    lp.integrality_[c] = HighsVarType::kInteger;
    lp.a_matrix_.start_[c + 1] = (HighsInt)lp.a_matrix_.index_.size() + 1;
    lp.a_matrix_.index_.push_back(7);
    lp.a_matrix_.value_.push_back(1.0);
  }
  return lp;
}

TEST_CASE("MIP-lagrangian-binding", "[highs_test_mip_solver]") {
  // Tight coupling row (binds at block optima): the Lagrangian loop
  // must run without changing the proven optimum.
  double objective_on = kHighsInf;
  for (bool lagrangian : {true, false}) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_lagrangian", lagrangian);
    highs.passModel(lagrangianTwoBlocks(60.0));
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (lagrangian)
      objective_on = objective;
    else
      REQUIRE(objective == objective_on);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("MIP-lagrangian-loose", "[highs_test_mip_solver]") {
  // Loose coupling row: block optima satisfy it, so the loop converges
  // immediately and may inject the composition as incumbent.
  double objective_on = kHighsInf;
  for (bool lagrangian : {true, false}) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("mip_abs_gap", 0);
    highs.setOptionValue("mip_lagrangian", lagrangian);
    highs.passModel(lagrangianTwoBlocks(95.0));
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    double objective = highs.getInfo().objective_function_value;
    REQUIRE(std::isfinite(objective));
    if (lagrangian)
      objective_on = objective;
    else
      REQUIRE(objective == objective_on);
    highs.resetGlobalScheduler(true);
  }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkControl.cpp
 * @brief
 */

#include "simplex/HEkk.h"

// Sliding window used to detect bimodal DSE solve densities that the
// smoothed-average based cost test cannot see: if a large fraction of
// the recent DSE solves were costly, Devex (whose per-iteration cost is
// O(pack)) is clearly preferable
constexpr HighsInt kCostlyDseWindowSize = 1000;
constexpr double kCostlyDseWindowCostlyFraction = 0.2;
constexpr HighsInt kCostlyDseWindowMinIterations = 2000;

void HEkk::initialiseControl() {
  // Copy tolerances from options
  info_.allow_dual_steepest_edge_to_devex_switch =
      options_->simplex_dual_edge_weight_strategy ==
      kSimplexEdgeWeightStrategyChoose;
  info_.dual_steepest_edge_weight_log_error_threshold =
      options_->dual_steepest_edge_weight_log_error_threshold;
  info_.control_iteration_count0 = iteration_count_;
  // Initialise the densities
  info_.col_aq_density = 0;
  info_.row_ep_density = 0;
  info_.row_ap_density = 0;
  info_.row_DSE_density = 0;
  info_.raw_row_DSE_density = 0;
  info_.costly_dse_in_window_ = 0;
  info_.costly_dse_ring_.assign(kCostlyDseWindowSize, false);
  info_.col_steepest_edge_density = 0;
  info_.col_basic_feasibility_change_density = 0;
  info_.row_basic_feasibility_change_density = 0;
  info_.col_BFRT_density = 0;
  info_.primal_col_density = 0;
  // Set the row_dual_density to 1 since it's assumed all costs are at
  // least perturbed from zero, if not initially nonzero
  info_.dual_col_density = 1;
  // Initialise the data used to determine the switch from DSE to
  // Devex
  info_.costly_DSE_frequency = 0;
  info_.num_costly_DSE_iteration = 0;
  info_.costly_DSE_measure = 0;
  info_.average_log_low_DSE_weight_error = 0;
  info_.average_log_high_DSE_weight_error = 0;
}

void HEkk::updateOperationResultDensity(const double local_density,
                                        double& density) const {
  density = (1 - kRunningAverageMultiplier) * density +
            kRunningAverageMultiplier * local_density;
}

void HEkk::assessDSEWeightError(const double computed_edge_weight,
                                const double updated_edge_weight) {
  // Compute the (relative) dual steepest edge weight error for
  // analysis and debugging
  edge_weight_error_ = std::fabs(updated_edge_weight - computed_edge_weight) /
                       std::max(1.0, computed_edge_weight);
  if (edge_weight_error_ > options_->dual_steepest_edge_weight_error_tolerance)
    highsLogDev(options_->log_options, HighsLogType::kInfo,
                "Dual steepest edge weight error is %g\n", edge_weight_error_);
  // Compute the relative deviation in the updated weight compared
  // with the computed weight
  double weight_relative_deviation;
  if (updated_edge_weight < computed_edge_weight) {
    // Updated weight is low
    weight_relative_deviation = computed_edge_weight / updated_edge_weight;
    info_.average_log_low_DSE_weight_error =
        0.99 * info_.average_log_low_DSE_weight_error +
        0.01 * log(weight_relative_deviation);
  } else {
    // Updated weight is correct or high
    weight_relative_deviation = updated_edge_weight / computed_edge_weight;
    info_.average_log_high_DSE_weight_error =
        0.99 * info_.average_log_high_DSE_weight_error +
        0.01 * log(weight_relative_deviation);
  }
}

bool HEkk::switchToDevex() {
  // Parameters controlling switch from DSE to Devex on cost
  const double kCostlyDseMeasureLimit = 1000.0;
  const double kCostlyDseMinimumDensity = 0.01;
  const double kCostlyDseFractionNumTotalIterationBeforeSwitch = 0.1;
  const double kCostlyDseFractionNumCostlyDseIterationBeforeSwitch = 0.05;
  bool switch_to_devex = false;
  // Firstly consider switching on the basis of NLA cost
  double costly_DSE_measure_denominator;
  costly_DSE_measure_denominator = max(
      max(info_.row_ep_density, info_.col_aq_density), info_.row_ap_density);

  // Track a sliding window of the raw per-iteration DSE solve density
  // relative to the other NLA solve densities: the smoothed average used
  // below cannot detect bimodal densities (many very cheap and many very
  // costly solves alternating), in which case the costly fraction is
  // invisible to the smoothed cost test
  if (info_.costly_dse_ring_.size() !=
      static_cast<size_t>(kCostlyDseWindowSize))
    info_.costly_dse_ring_.assign(kCostlyDseWindowSize, false);
  const HighsInt ring_index = iteration_count_ % kCostlyDseWindowSize;
  double raw_costly_DSE_measure = 0;
  if (costly_DSE_measure_denominator > 0 && info_.raw_row_DSE_density > 0) {
    raw_costly_DSE_measure =
        info_.raw_row_DSE_density / costly_DSE_measure_denominator;
    raw_costly_DSE_measure *= raw_costly_DSE_measure;
  }
  const bool costly_dse_iteration_raw =
      raw_costly_DSE_measure > kCostlyDseMeasureLimit &&
      info_.raw_row_DSE_density > kCostlyDseMinimumDensity;
  info_.costly_dse_in_window_ -= info_.costly_dse_ring_[ring_index];
  info_.costly_dse_ring_[ring_index] = costly_dse_iteration_raw;
  info_.costly_dse_in_window_ += costly_dse_iteration_raw;
  const HighsInt local_iteration_count_window =
      iteration_count_ - info_.control_iteration_count0;
  if (local_iteration_count_window >= kCostlyDseWindowMinIterations) {
    const HighsInt window_count =
        std::min(local_iteration_count_window, kCostlyDseWindowSize);
    switch_to_devex =
        info_.allow_dual_steepest_edge_to_devex_switch &&
        info_.costly_dse_in_window_ >
            kCostlyDseWindowCostlyFraction * window_count;
    if (switch_to_devex) {
      highsLogDev(options_->log_options, HighsLogType::kInfo,
                  "Switch from DSE to Devex after %" HIGHSINT_FORMAT
                  " costly DSE iterations of the last %" HIGHSINT_FORMAT
                  " with raw density %11.4g\n",
                  info_.costly_dse_in_window_, window_count,
                  info_.raw_row_DSE_density);
    }
  }

  if (!switch_to_devex) {
    // Secondly consider switching on the basis of weight accuracy
    double local_measure = info_.average_log_low_DSE_weight_error +
                           info_.average_log_high_DSE_weight_error;
    double local_threshold =
        info_.dual_steepest_edge_weight_log_error_threshold;
    switch_to_devex = info_.allow_dual_steepest_edge_to_devex_switch &&
                      local_measure > local_threshold;
    if (switch_to_devex) {
      highsLogDev(options_->log_options, HighsLogType::kInfo,
                  "Switch from DSE to Devex with log error measure of %g > "
                  "%g = threshold\n",
                  local_measure, local_threshold);
    }
  }
  return switch_to_devex;
}

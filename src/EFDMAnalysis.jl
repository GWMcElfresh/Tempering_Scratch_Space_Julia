module EFDMAnalysis

using Pigeons, Distributions, Random, Statistics, Printf, LinearAlgebra
using CSV, DataFrames, TOML
using MCMCChains
import LogDensityProblems, ForwardDiff
using SpecialFunctions: loggamma

# Re-export key public API
export EFDMLogPotential, EFDMReference
export efdm_n_params, simulate_efdm
export ConvergenceSummary, convergence_summary, convergence_assessment
export extract_target_chain_indices, extract_parameter_draws

# Core math (pure functions)
include("core.jl")

# Pigeons.jl target and reference distributions
include("targets.jl")

# Post-sampling diagnostics
include("diagnostics.jl")

# Plotting and visualization
include("plotting.jl")

# Auto-tune: covariance, shrinkage, empirical Bayes
include("autotune.jl")

# Re-export auto-tune API
export shrinkage_report, shrinkage_report_str
export posterior_logit_w_covariance, posterior_alr_p_covariance
export posterior_mu_covariance, posterior_mu_covariance_cond
export auto_tune_round, run_auto_tune
export plot_covariance_heatmap, plot_shrinkage_barchart, plot_all_covariances

end # module

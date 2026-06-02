module EFDMAnalysis

using Pigeons, Distributions, Random, Statistics, Printf
using CSV, DataFrames, TOML
using MCMCChains
import LogDensityProblems, ForwardDiff
using SpecialFunctions: loggamma

# Re-export key public API
export EFDMLogPotential, EFDMReference
export efdm_n_params, simulate_efdm
export ConvergenceSummary, convergence_summary, convergence_assessment
export extract_target_chain_indices, extract_parameter_draws
export load_data, build_design_matrix, merge_rare_types

# Core math (pure functions)
include("core.jl")

# Pigeons.jl target and reference distributions
include("targets.jl")

# Post-sampling diagnostics
include("diagnostics.jl")

# Plotting and visualization
include("plotting.jl")

end # module

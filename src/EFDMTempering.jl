"""
EFDMTempering.jl

A Julia wrapper package that provides a high-level interface for fitting count
composition models (from the EFDMReg framework) using Pigeons.jl's parallel
tempering MCMC algorithm. Supports rugged posterior geometries via simulated
tempering.

Key models supported:
- Multinomial regression
- Dirichlet-Multinomial (DM) regression
- Flexible Dirichlet-Multinomial (FDM) regression
- Extended Flexible Dirichlet-Multinomial (EFDM) regression (per-category weights)

Usage:
    using BridgeStan   # must be loaded before calling Stan targets
    using EFDMTempering

    # Fit with a bundled EFDM model using a CSV data file
    result = fit_efdm("path/to/data.csv")

    # Fit any Stan model with CSV data
    result = fit_stan_with_pigeons("path/to/model.stan", "path/to/data.csv")

    # Extract posterior samples as an MCMCChains.Chains object
    using MCMCChains
    chains = get_chains(result)
"""
module EFDMTempering

using CSV
using DataFrames
using JSON
using Pigeons

export fit_stan_with_pigeons,
       fit_efdm,
       fit_fdm,
       fit_dm,
       fit_multinomial,
       prepare_efdm_data,
       get_chains,
       stan_model_path,
       BUNDLED_MODELS

# ── Bundled Stan model registry ────────────────────────────────────────────────

"""
    BUNDLED_MODELS

A named tuple mapping model names to their bundled `.stan` file paths.

Models:
- `:multinomial`  – Multinomial regression
- `:dm`           – Dirichlet-Multinomial regression
- `:fdm`          – Flexible Dirichlet-Multinomial regression
- `:efdm`         – Extended Flexible Dirichlet-Multinomial regression
"""
const BUNDLED_MODELS = (
    multinomial = joinpath(@__DIR__, "..", "stan_models", "Multinomial.stan"),
    dm          = joinpath(@__DIR__, "..", "stan_models", "DM.stan"),
    fdm         = joinpath(@__DIR__, "..", "stan_models", "FDM2.stan"),
    efdm        = joinpath(@__DIR__, "..", "stan_models", "EFDM_hyper_w.stan"),
)

"""
    stan_model_path(model_name::Symbol) -> String

Return the absolute path to a bundled Stan model file.

# Arguments
- `model_name`: one of `:multinomial`, `:dm`, `:fdm`, or `:efdm`
"""
function stan_model_path(model_name::Symbol)
    p = get(BUNDLED_MODELS, model_name, nothing)
    p === nothing && throw(ArgumentError(
        "Unknown model '$model_name'. Valid names: $(keys(BUNDLED_MODELS))"
    ))
    abspath(p)
end

# ── CSV → Stan data helpers ───────────────────────────────────────────────────

"""
    load_csv(csv_path::AbstractString) -> DataFrame

Load a CSV file and return it as a `DataFrame`.
"""
function load_csv(csv_path::AbstractString)
    isfile(csv_path) || throw(ArgumentError("CSV file not found: $csv_path"))
    CSV.read(csv_path, DataFrame)
end

"""
    dict_to_json(data::Dict) -> String

Serialise a data dictionary to a JSON string suitable for Stan / BridgeStan.
Arrays and matrices are serialised in row-major order as nested arrays, which
matches what Stan expects in JSON data files.
"""
function dict_to_json(data::Dict)
    JSON.json(data)
end

"""
    prepare_efdm_data(csv_path; response_cols, covariate_cols,
                      sd_prior, w_hyper, n_trials) -> String (JSON)

Build a JSON data string for the EFDM / FDM / DM models from a CSV file.

# Arguments
- `csv_path`       : path to the CSV file
- `response_cols`  : column names for the count response columns
                     (default: all columns whose names start with `"Y"`)
- `covariate_cols` : column names for the covariate columns
                     (default: all remaining columns; an intercept is prepended)
- `sd_prior`       : prior standard deviation for `beta` (default: `50.0`)
- `w_hyper`        : hyperparameter vector for `w_norm`; length must equal `D`
                     (default: `ones(D)`)
- `n_trials`       : integer vector of trial counts (one per row).
                     If `nothing`, row-sums of `response_cols` are used.

# Returns
A JSON string compatible with `Pigeons.StanLogPotential`.
"""
function prepare_efdm_data(
    csv_path::AbstractString;
    response_cols  = nothing,
    covariate_cols = nothing,
    sd_prior::Real = 50.0,
    w_hyper        = nothing,
    n_trials       = nothing,
)
    df = load_csv(csv_path)
    colnames = names(df)

    # ── Identify response columns ──────────────────────────────────────────
    if response_cols === nothing
        response_cols = filter(c -> startswith(uppercase(c), "Y"), colnames)
        isempty(response_cols) && throw(ArgumentError(
            "Could not auto-detect response columns. " *
            "Please supply `response_cols` explicitly."
        ))
    end

    Y = Matrix{Int}(df[:, response_cols])
    N, D = size(Y)

    # ── Identify covariate columns ─────────────────────────────────────────
    if covariate_cols === nothing
        covariate_cols = setdiff(colnames, response_cols)
    end

    if isempty(covariate_cols)
        X = ones(N, 1)
    else
        X_raw = Matrix{Float64}(df[:, covariate_cols])
        X = hcat(ones(N, 1), X_raw)
    end
    K = size(X, 2)

    # ── Trial counts ───────────────────────────────────────────────────────
    n_vec = if n_trials === nothing
        vec(sum(Y, dims=2))
    else
        Int.(n_trials)
    end

    # ── Hyperparameters ────────────────────────────────────────────────────
    w_hyper_vec = w_hyper === nothing ? ones(D) : Float64.(w_hyper)
    length(w_hyper_vec) == D || throw(ArgumentError(
        "w_hyper must have length D=$D, got $(length(w_hyper_vec))"
    ))

    # Stan expects 2D arrays as arrays-of-arrays (row-major)
    data_dict = Dict{String,Any}(
        "N"        => N,
        "D"        => D,
        "K"        => K,
        "n"        => n_vec,
        "X"        => [X[i,:] for i in 1:N],
        "Y"        => [Y[i,:] for i in 1:N],
        "sd_prior" => Float64(sd_prior),
        "w_hyper"  => w_hyper_vec,
    )

    dict_to_json(data_dict)
end

# ── Core fitting function ─────────────────────────────────────────────────────

"""
    fit_stan_with_pigeons(stan_file, csv_file;
                          data_prep_fn      = prepare_efdm_data,
                          data_prep_kwargs  = (;),
                          n_rounds          = 12,
                          n_chains          = 8,
                          pigeons_kwargs...)
        -> Pigeons.PT

Run Pigeons.jl parallel tempering on the Stan model at `stan_file` using data
loaded from `csv_file`.

!!! note
    `BridgeStan` must be loaded in the calling session before this function is
    called: `using BridgeStan`.

# Arguments
- `stan_file`        : path to a `.stan` model file (absolute or relative)
- `csv_file`         : path to a CSV data file
- `data_prep_fn`     : function `(csv_path; kwargs...) -> JSON_string` that
                       converts the CSV into a Stan data JSON string
                       (default: `prepare_efdm_data`)
- `data_prep_kwargs` : keyword arguments forwarded to `data_prep_fn`
- `n_rounds`         : number of Pigeons annealing rounds (default: `12`)
- `n_chains`         : number of parallel tempering chains (default: `8`)
- `pigeons_kwargs`   : additional keyword arguments forwarded to `pigeons()`

# Returns
A `Pigeons.PT` object. Use `get_chains(result)` to extract an `MCMCChains.Chains`.
"""
function fit_stan_with_pigeons(
    stan_file::AbstractString,
    csv_file::AbstractString;
    data_prep_fn     = prepare_efdm_data,
    data_prep_kwargs = (;),
    n_rounds::Int    = 12,
    n_chains::Int    = 8,
    pigeons_kwargs...,
)
    isfile(stan_file) || throw(ArgumentError("Stan model not found: $stan_file"))
    isfile(csv_file)  || throw(ArgumentError("CSV data file not found: $csv_file"))

    # Build JSON data string for Stan / BridgeStan
    stan_data_json = data_prep_fn(csv_file; data_prep_kwargs...)

    # StanLogPotential(stan_file, data_json_string [, extra_information])
    target = Pigeons.StanLogPotential(abspath(stan_file), stan_data_json)

    pt = pigeons(
        target      = target,
        n_rounds    = n_rounds,
        n_chains    = n_chains,
        record      = [traces, round_trip, log_sum_ratio],
        pigeons_kwargs...,
    )

    return pt
end

# ── Convenience wrappers for each bundled model ───────────────────────────────

"""
    fit_efdm(csv_file; kwargs...) -> Pigeons.PT

Fit the Extended Flexible Dirichlet-Multinomial (EFDM) model using Pigeons.jl.
This is the most flexible model in the EFDMReg family: it supports per-category
mixing weights and can represent negative covariance structures.

!!! note
    `BridgeStan` must be loaded before calling: `using BridgeStan`.

See `fit_stan_with_pigeons` for a full description of keyword arguments.
"""
function fit_efdm(csv_file::AbstractString; kwargs...)
    fit_stan_with_pigeons(stan_model_path(:efdm), csv_file; kwargs...)
end

"""
    fit_fdm(csv_file; kwargs...) -> Pigeons.PT

Fit the Flexible Dirichlet-Multinomial (FDM) model using Pigeons.jl.

See `fit_stan_with_pigeons` for a full description of keyword arguments.
"""
function fit_fdm(csv_file::AbstractString; kwargs...)
    fit_stan_with_pigeons(stan_model_path(:fdm), csv_file; kwargs...)
end

"""
    fit_dm(csv_file; kwargs...) -> Pigeons.PT

Fit the Dirichlet-Multinomial (DM) regression model using Pigeons.jl.

See `fit_stan_with_pigeons` for a full description of keyword arguments.
"""
function fit_dm(csv_file::AbstractString; kwargs...)
    fit_stan_with_pigeons(stan_model_path(:dm), csv_file; kwargs...)
end

"""
    fit_multinomial(csv_file; kwargs...) -> Pigeons.PT

Fit the Multinomial regression model using Pigeons.jl.

See `fit_stan_with_pigeons` for a full description of keyword arguments.
"""
function fit_multinomial(csv_file::AbstractString; kwargs...)
    fit_stan_with_pigeons(stan_model_path(:multinomial), csv_file;
        data_prep_fn = _prepare_multinomial_data, kwargs...)
end

# Multinomial model uses a slightly different data block (no n or w_hyper)
function _prepare_multinomial_data(
    csv_path::AbstractString;
    response_cols  = nothing,
    covariate_cols = nothing,
    sd_prior::Real = 50.0,
    kwargs...,
)
    df = load_csv(csv_path)
    colnames = names(df)

    if response_cols === nothing
        response_cols = filter(c -> startswith(uppercase(c), "Y"), colnames)
        isempty(response_cols) && throw(ArgumentError(
            "Could not auto-detect response columns. " *
            "Please supply `response_cols` explicitly."
        ))
    end

    Y = Matrix{Int}(df[:, response_cols])
    N, D = size(Y)

    if covariate_cols === nothing
        covariate_cols = setdiff(colnames, response_cols)
    end

    if isempty(covariate_cols)
        X = ones(N, 1)
    else
        X_raw = Matrix{Float64}(df[:, covariate_cols])
        X = hcat(ones(N, 1), X_raw)
    end
    K = size(X, 2)

    data_dict = Dict{String,Any}(
        "N"        => N,
        "D"        => D,
        "K"        => K,
        "X"        => [X[i,:] for i in 1:N],
        "Y"        => [Y[i,:] for i in 1:N],
        "sd_prior" => Float64(sd_prior),
    )

    dict_to_json(data_dict)
end

# ── Result extraction helpers ─────────────────────────────────────────────────

"""
    get_chains(pt::Pigeons.PT) -> MCMCChains.Chains

Extract posterior samples from a completed Pigeons.jl run as an
`MCMCChains.Chains` object for downstream diagnostics and summaries.

Requires `MCMCChains` to be loaded in the calling session: `using MCMCChains`.
"""
function get_chains(pt)
    # Pigeons provides Chains conversion via its MCMCChains extension when
    # MCMCChains is loaded in the current session
    return Pigeons.Chains(pt)
end

end # module EFDMTempering

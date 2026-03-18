# EFDMTempering.jl

A Julia package that wraps the **EFDMReg** count-composition models with
**Pigeons.jl** parallel tempering MCMC, making it easy to fit Dirichlet-family
regression models on rugged posterior geometries.

## Background

### EFDMReg models

[EFDMReg](https://github.com/robertoascari/EFDMReg) implements a family of
Bayesian regression models for compositional count data:

| Symbol | Model | Key feature |
|--------|-------|-------------|
| `:multinomial` | Multinomial regression | Baseline |
| `:dm` | Dirichlet-Multinomial | Overdispersion |
| `:fdm` | Flexible Dirichlet-Multinomial | Single global mixing weight |
| `:efdm` | Extended FDM | Per-category mixing weights – allows **negative covariance** |

All models are defined as Stan programs in the `stan_models/` directory and are
identical to the originals in EFDMReg.

### Pigeons.jl

[Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl) provides
parallel/simulated tempering that efficiently explores multimodal or
highly-curved posteriors that trip up vanilla HMC.

## Installation

```julia
# From the Julia REPL (with the repository as the active project):
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick start

```julia
using EFDMTempering

# Fit the EFDM model on a CSV of count data
# Columns whose names start with "Y" are treated as response categories;
# all other columns become covariates (an intercept is prepended automatically).
pt = fit_efdm("path/to/data.csv";
              n_rounds = 12,   # annealing rounds
              n_chains = 8)    # parallel tempering chains

# Extract posterior samples as an MCMCChains.Chains object
using MCMCChains
chains = get_chains(pt)
println(chains)
```

### Other bundled models

```julia
pt_dm   = fit_dm("data.csv")
pt_fdm  = fit_fdm("data.csv")
pt_mult = fit_multinomial("data.csv")
```

### Bring your own Stan model

```julia
pt = fit_stan_with_pigeons(
    "path/to/my_model.stan",
    "path/to/data.csv";
    n_rounds = 10,
    n_chains = 4,
)
```

## CSV format

The CSV must contain:

- **Response columns** – integer counts; by default any column whose name
  starts with `Y` (case-insensitive). Pass `response_cols = ["col1", ...]` to
  override.
- **Covariate columns** – numeric covariates. All non-response columns are used
  by default. An intercept column is prepended automatically. Pass
  `covariate_cols = [...]` to override.

See `data/example_counts.csv` for a working example.

### Data preparation options

```julia
pt = fit_efdm("data.csv";
    data_prep_kwargs = (
        response_cols  = ["Y1","Y2","Y3","Y4"],
        covariate_cols = ["x1","x2"],
        sd_prior       = 10.0,
        w_hyper        = [1.0, 1.0, 1.0, 1.0],
    ),
)
```

## Docker / CI

The repository ships with a `Dockerfile` and a GitHub Actions workflow
(`.github/workflows/ci.yml`) that uses the
[dockerDependencies](https://github.com/GWMcElfresh/dockerDependencies)
reusable workflow for staged dependency caching. The base dependency image is
rebuilt monthly; the runtime image is rebuilt on every push.

## API reference

| Function | Description |
|----------|-------------|
| `fit_efdm(csv; kwargs...)` | Fit EFDM model (recommended) |
| `fit_fdm(csv; kwargs...)` | Fit FDM model |
| `fit_dm(csv; kwargs...)` | Fit DM model |
| `fit_multinomial(csv; kwargs...)` | Fit Multinomial model |
| `fit_stan_with_pigeons(stan, csv; kwargs...)` | Fit any Stan model |
| `prepare_efdm_data(csv; kwargs...)` | Build Stan data dict from CSV |
| `stan_model_path(sym)` | Path to a bundled `.stan` file |
| `get_chains(pt)` | Extract `MCMCChains.Chains` from Pigeons result |
| `BUNDLED_MODELS` | Named tuple of bundled model paths |

## References

- Ascari, R. & Migliorati, S. (2021). *A new regression model for overdispersed
  binomial data accounting for outliers and an excess of zeros.*
  Statistics in Medicine.
- Surjanovic, N. et al. (2023). *Pigeons.jl: Distributed sampling from
  intractable distributions.* arXiv:2308.09769.
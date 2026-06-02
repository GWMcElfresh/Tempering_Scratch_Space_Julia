# =============================================================================
# EFDM Plotting — Conditional Effects and Diagnostic Visualizations
# =============================================================================
# Plotting functions for EFDM posterior exploration. Each function accepts
# the Pigeons PT result and produces either:
#   - A PlotlyJS HTML file (preferred, interactive)
#   - A CSV export (portable, for external plotting in R/Python)
#
# PlotlyJS is an optional dependency. If not installed, functions will write
# data CSVs and print instructions for plotting.
# =============================================================================

# Try to load PlotlyJS; if unavailable, set a flag
const _HAS_PLOTLY = try
    using PlotlyJS
    true
catch
    false
end

"""
    _maybe_plotly(fig, filepath)

Save a PlotlyJS figure to HTML, or print a message if PlotlyJS is unavailable.
"""
function _maybe_plotly(fig, filepath::String)
    if _HAS_PLOTLY
        savehtml(fig, filepath)
        @info "Saved plot: $filepath"
    else
        csv_path = splitext(filepath)[1] * "_data.csv"
        @info "PlotlyJS not available. Skipping $filepath"
        @info "  Install PlotlyJS via `using Pkg; Pkg.add(\"PlotlyJS\")`"
        @info "  Data for external plotting written to $csv_path (if applicable)"
    end
end

"""
    plot_aplus_posterior(pt, D, K; output_dir=".", filename="aplus_posterior.html")

Histogram of aplus (overdispersion) from posterior draws.
"""
function plot_aplus_posterior(pt, D::Int, K::Int;
                              output_dir::String=".", filename::String="aplus_posterior.html")
    samples = sample_array(pt)
    tci = extract_target_chain_indices(pt)
    idx_aplus = (D - 1) * K + 1

    raw_log_aplus = extract_parameter_draws(samples, idx_aplus, tci)
    aplus_draws = exp.(min.(raw_log_aplus, 700.0))
    aplus_mean = mean(aplus_draws)
    aplus_median = median(aplus_draws)
    aplus_ci = quantile(aplus_draws, [0.025, 0.975])

    outpath = joinpath(output_dir, filename)
    csv_path = joinpath(output_dir, "aplus_posterior_data.csv")

    # Write CSV data
    df = DataFrame(aplus = aplus_draws)
    CSV.write(csv_path, df)
    @info "Wrote aplus posterior draws: $csv_path  (mean=$(round(aplus_mean;digits=1)))"

    if _HAS_PLOTLY
        hist = PlotlyJS.histogram(; x=aplus_draws, nbinsx=80,
                                   name="aplus",
                                   marker_color="steelblue",
                                   opacity=0.75)
        vline = PlotlyJS.shape(;
            type="line", x0=aplus_mean, y0=0,
            x1=aplus_mean, y1=1, yref="paper",
            line=PlotlyJS.attr(color="red", width=2, dash="dash")
        )
        layout = PlotlyJS.Layout(;
            title="Posterior of aplus (overdispersion)",
            xaxis_title="aplus",
            yaxis_title="Count",
            shapes=[vline],
            annotations=[
                PlotlyJS.attr(
                    x=0.95, y=0.95, xref="paper", yref="paper",
                    text=@sprintf("mean=%.1f<br>median=%.1f<br>95%%CI=[%.1f, %.1f]",
                                  aplus_mean, aplus_median, aplus_ci[1], aplus_ci[2]),
                    showarrow=false, font_size=12,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            ]
        )
        fig = PlotlyJS.plot(hist, layout)
        _maybe_plotly(fig, outpath)
    end

    return (mean=aplus_mean, median=aplus_median, ci=aplus_ci)
end

"""
    plot_conditional_effects(pt, D, K, X, Y, covariate_names;
                             output_dir=".", filename="conditional_effects.html",
                             n_grid=50)

Plot expected cell-type proportions as a function of each continuous covariate
(holding others at their mean).

For each covariate c:
  - Vary the covariate over its observed range (n_grid points)
  - Hold all other covariates at their column means
  - Compute expected proportions at the posterior mean of β
  - Plot a line per cell type

Returns a DataFrame with all grid predictions (for external plotting).
"""
function plot_conditional_effects(pt, D::Int, K::Int,
                                  X::Matrix{Float64},
                                  Y::Matrix{Float64},
                                  covariate_names::AbstractVector{<:AbstractString};
                                  output_dir::String=".",
                                  filename::String="conditional_effects.html",
                                  n_grid::Int=50)
    N_obs = size(X, 1)
    samples = sample_array(pt)
    tci = extract_target_chain_indices(pt)

    n_params = size(samples, 2)
    n_iter = size(samples, 1)
    n_cold = length(tci)

    # Posterior mean of θ (across all cold chain draws)
    θ_mean = zeros(n_params)
    for p in 1:n_params
        draws = extract_parameter_draws(samples, p, tci)
        θ_mean[p] = mean(draws)
    end

    # Unpack to get mean β
    beta_mean, aplus_mean, p_mean, w_mean = unpack_efdm(θ_mean, D, K)

    # For each covariate (skip intercept = column 1), compute grid effects
    cov_indices = 2:K
    if length(cov_indices) == 0
        @warn "No non-intercept covariates to plot. Skipping conditional effects."
        return DataFrame()
    end

    all_rows = []
    x_means = vec(mean(X; dims=1))

    for cov_idx in cov_indices
        cov_name = cov_idx ≤ length(covariate_names) ? covariate_names[cov_idx] : "Covariate_$(cov_idx)"
        x_min = minimum(X[:, cov_idx])
        x_max = maximum(X[:, cov_idx])
        x_grid = range(x_min, x_max; length=n_grid)

        mu_grid = zeros(n_grid, D)
        for (gi, x_val) in enumerate(x_grid)
            x_row = copy(x_means)
            x_row[cov_idx] = x_val
            softmax_linear_row!(@view(mu_grid[gi, :]), x_row, beta_mean)
        end

        for d in 1:D
            for gi in 1:n_grid
                push!(all_rows, (
                    covariate=cov_name,
                    x=x_grid[gi],
                    cell_type="CT$(d-1)",
                    expected_proportion=mu_grid[gi, d]
                ))
            end
        end
    end

    df = DataFrame(all_rows)

    # Write CSV
    csv_path = joinpath(output_dir, "conditional_effects_data.csv")
    CSV.write(csv_path, df)
    @info "Wrote conditional effects data: $csv_path"

    if _HAS_PLOTLY
        traces = []
        for d in 1:D
            for cov_name in unique(df.covariate)
                subset_df = df[(df.cell_type .== "CT$(d-1)") .& (df.covariate .== cov_name), :]
                if nrow(subset_df) > 0
                    tr = PlotlyJS.scatter(;
                        x=subset_df.x, y=subset_df.expected_proportion,
                        mode="lines",
                        name="CT$(d-1)",
                        line=PlotlyJS.attr(width=2)
                    )
                    push!(traces, tr)
                end
            end
        end

        unique_covs = unique(df.covariate)
        if length(unique_covs) == 1
            layout = PlotlyJS.Layout(;
                title="Expected proportions (posterior mean β)",
                xaxis_title=unique_covs[1],
                yaxis_title="Expected proportion",
                legend_title="Cell type"
            )
            fig = PlotlyJS.plot(traces, layout)
            _maybe_plotly(fig, joinpath(output_dir, filename))
        else
            # One subplot per covariate
            for cov_name in unique_covs
                cov_traces = [tr for tr in traces if occursin(cov_name, string(tr["name"]))]
                cov_traces = []
                for d in 1:D
                    subset_df = df[(df.cell_type .== "CT$(d-1)") .& (df.covariate .== cov_name), :]
                    if nrow(subset_df) > 0
                        tr = PlotlyJS.scatter(;
                            x=subset_df.x, y=subset_df.expected_proportion,
                            mode="lines", name="CT$(d-1)"
                        )
                        push!(cov_traces, tr)
                    end
                end
                layout = PlotlyJS.Layout(;
                    title="Effect of $cov_name on proportions",
                    xaxis_title=cov_name,
                    yaxis_title="Expected proportion"
                )
                fig = PlotlyJS.plot(cov_traces, layout)
                _maybe_plotly(fig, joinpath(output_dir, "effect_$(cov_name).html"))
            end
        end
    end

    return df
end

"""
    plot_rhat_summary(pt; output_dir=".", filename="rhat_summary.html")

Ordered bar chart of Rhat values with a reference line at 1.05.

Writes CSV data in all cases, and an interactive HTML if PlotlyJS is available.
"""
function plot_rhat_summary(pt; output_dir::String=".", filename::String="rhat_summary.html")
    chain_obj = Chains(pt)
    rhat_vals = rhat(chain_obj)

    rhat_vec = if rhat_vals isa DataFrame
        Float64[skipmissing(rhat_vals.nt.rhat)...]
    else
        Float64[skipmissing(rhat_vals)...]
    end

    param_names = if rhat_vals isa DataFrame
        [string(n) for n in rhat_vals.nt.parameters]
    else
        ["Param_$i" for i in 1:length(rhat_vec)]
    end

    # Sort by Rhat value
    order = sortperm(rhat_vec)
    rhat_sorted = rhat_vec[order]
    names_sorted = param_names[order]

    df = DataFrame(param=names_sorted, rhat=rhat_sorted)
    csv_path = joinpath(output_dir, "rhat_data.csv")
    CSV.write(csv_path, df)
    @info "Wrote Rhat data: $csv_path"

    if _HAS_PLOTLY
        colors = [v > 1.05 ? "crimson" : (v > 1.01 ? "orange" : "steelblue") for v in rhat_sorted]
        bar = PlotlyJS.bar(;
            x=names_sorted, y=rhat_sorted,
            marker_color=colors,
            name="Rhat"
        )
        hline = PlotlyJS.shape(;
            type="line", x0=-0.5, y0=1.05,
            x1=length(rhat_sorted) - 0.5, y1=1.05,
            line=PlotlyJS.attr(color="red", width=2, dash="dash"),
            yref="y"
        )
        layout = PlotlyJS.Layout(;
            title="Rhat values by parameter (sorted)",
            xaxis_title="Parameter",
            yaxis_title="Rhat",
            shapes=[hline],
            xaxis=PlotlyJS.attr(tickangle=45)
        )
        fig = PlotlyJS.plot(bar, layout)
        _maybe_plotly(fig, joinpath(output_dir, filename))
    end

    return df
end

"""
    plot_traces(pt, param_indices, param_names;
                output_dir=".", filename="traces.html")

Trace plots for selected parameters (one subplot per parameter).
Each subplot shows the trace of all cold chains.

`param_indices` — vector of parameter indices (1-indexed columns of sample_array)
`param_names` — vector of display names, same length as param_indices
"""
function plot_traces(pt, param_indices::Vector{Int},
                     param_names::AbstractVector{<:AbstractString};
                     output_dir::String=".", filename::String="traces.html")
    samples = sample_array(pt)
    tci = extract_target_chain_indices(pt)
    n_iter = size(samples, 1)
    n_params = length(param_indices)

    # Write CSV traces
    rows = []
    for (pi, p_idx) in enumerate(param_indices)
        name = pi ≤ length(param_names) ? param_names[pi] : "Param_$p_idx"
        for ci in tci
            for it in 1:n_iter
                push!(rows, (
                    parameter=name,
                    iteration=it,
                    chain=ci,
                    value=samples[it, p_idx, ci]
                ))
            end
        end
    end
    df = DataFrame(rows)
    csv_path = joinpath(output_dir, "trace_data.csv")
    CSV.write(csv_path, df)
    @info "Wrote trace data: $csv_path"

    if _HAS_PLOTLY
        for (pi, p_idx) in enumerate(param_indices)
            name = pi ≤ length(param_names) ? param_names[pi] : "Param_$p_idx"
            tr = PlotlyJS.scatter(;
                x=1:n_iter, y=samples[:, p_idx, :],
                mode="lines",
                name="$name (all cold chains)",
                line=PlotlyJS.attr(width=1, opacity=0.6)
            )
            layout = PlotlyJS.Layout(;
                title="Trace: $name",
                xaxis_title="Iteration",
                yaxis_title=name
            )
            fig = PlotlyJS.plot(tr, layout)
            _maybe_plotly(fig, joinpath(output_dir, "trace_$(name).html"))
        end
    end

    return df
end

"""
    plot_all_diagnostics(pt, D, K, X, Y, covariate_names;
                         output_dir=".")

Convenience function that runs all diagnostic plots and saves them to
`output_dir`. Also returns a text summary.

Intended for use by the runner script.
"""
function plot_all_diagnostics(pt, D::Int, K::Int,
                              X::Matrix{Float64}, Y::Matrix{Float64},
                              covariate_names::AbstractVector{<:AbstractString};
                              output_dir::String="efdm_output")
    mkpath(output_dir)

    # 1. Convergence report
    summary = convergence_summary(pt)
    report = convergence_report_str(summary)
    println(report)

    # Save report to file
    report_path = joinpath(output_dir, "convergence_report.txt")
    write(report_path, report)
    @info "Saved convergence report: $report_path"

    # 2. Posterior summary
    param_names = generate_param_names(D, K)
    samples = sample_array(pt)
    tci = extract_target_chain_indices(pt)
    post_df = posterior_summary_table(samples, param_names, tci)
    post_csv = joinpath(output_dir, "posterior_summary.csv")
    CSV.write(post_csv, post_df)
    @info "Saved posterior summary: $post_csv"

    # 3. Plots
    plot_aplus_posterior(pt, D, K; output_dir=output_dir)
    plot_conditional_effects(pt, D, K, X, Y, covariate_names; output_dir=output_dir)
    plot_rhat_summary(pt; output_dir=output_dir)

    # Trace plots for key parameters: log(aplus), first few β, first few w
    key_params = Int[ (D-1)*K + 1 ]  # log(aplus)
    key_names = String["log_aplus"]
    n_beta = (D-1) * K
    for i in 1:min(n_beta, 4)
        push!(key_params, i)
        push!(key_names, "β[$i]")
    end
    plot_traces(pt, key_params, key_names; output_dir=output_dir)

    return summary
end

"""
plot_h2o_percentiles.jl
=======================
Side-by-side bound H₂O fields for the posterior ensemble percentiles:
  p05 (driest) | p50 (median) | p95 (wettest)

Companion to plot_ensemble_pseudosection.jl. Where that script collapses the
spread into a single (p95 − p05) panel, this one shows the three percentile
fields directly so you can see *where* the wet and dry ends of the ensemble
diverge — i.e. how far the lawsonite-out (dehydration) front migrates in P-T
between the driest and wettest bulk compositions.

Key point: all three panels use a SHARED color scale (same clims, same
colormap). Without that they auto-scale independently and are not comparable.

Usage
-----
  julia plot_h2o_percentiles.jl

Edit the PATHS block to match your setup.
"""

using Plots, Statistics

# =============================================================================
# PATHS  — edit these (mirrors plot_ensemble_pseudosection.jl)
# =============================================================================
const H2O_LOOKUP_DIR = "/Users/hchoi342/Documents/Archean-OC/h2o_lookup"
# const SCENARIO     = "homogeneous_crust"
const SCENARIO       = "layered_cumulate_lower_crust"

p05_path = joinpath(H2O_LOOKUP_DIR, "v2_h2o_p05_$(SCENARIO).csv")
p50_path = joinpath(H2O_LOOKUP_DIR, "v2_h2o_p50_$(SCENARIO).csv")
p95_path = joinpath(H2O_LOOKUP_DIR, "v2_h2o_p95_$(SCENARIO).csv")

const OUTPATH = "/Users/hchoi342/Documents/Archean-OC/pseudosection/h2o_percentiles_$(SCENARIO).png"

# Set true to also drop a (p95 − p05) panel into the figure (4 panels total)
const SHOW_RANGE = false

# =============================================================================
# HELPER: read CR-only CSV (vanKeken / Perple_X pipeline format)
# Identical to plot_ensemble_pseudosection.jl so the parse is consistent.
# =============================================================================
function read_cr_csv(path)
    raw    = read(path, String)
    rows   = split(raw, '\r')
    rows   = filter(r -> !isempty(strip(r)), rows)
    header = split(rows[1], ',')[2:end]          # skip leading empty cell
    n_rows = length(rows) - 1
    n_cols = length(header)
    data   = fill(NaN, n_rows, n_cols)
    p_vals = Float64[]
    for (i, row) in enumerate(rows[2:end])
        isempty(strip(row)) && continue
        cells = split(row, ',')
        push!(p_vals, parse(Float64, cells[1]))
        for (j, c) in enumerate(cells[2:end])
            data[i, j] = isempty(strip(c)) ? NaN : parse(Float64, c)
        end
    end
    t_vals = parse.(Float64, header)
    return p_vals, t_vals, data
end

# =============================================================================
# LOAD
# =============================================================================
println("Loading percentile CSVs for scenario: ", SCENARIO)
p_vals, t_vals, h2o_p05 = read_cr_csv(p05_path)
_,      _,      h2o_p50 = read_cr_csv(p50_path)
_,      _,      h2o_p95 = read_cr_csv(p95_path)

t_C = t_vals .- 273.15   # K → °C for axis labels (P stays in GPa as stored)

for (name, M) in (("p05", h2o_p05), ("p50", h2o_p50), ("p95", h2o_p95))
    finite = filter(!isnan, M)
    println("  $name range : ",
        round(minimum(finite), digits=3), " – ",
        round(maximum(finite), digits=3), " wt%")
end

# =============================================================================
# SHARED COLOR SCALE  — the whole point of this comparison
# Common clims across p05/p50/p95 so a pixel's color means the same thing
# in every panel. Anchored at 0; top set by the global (p95) maximum.
# =============================================================================
vmax = maximum(filter(!isnan, vcat(vec(h2o_p05), vec(h2o_p50), vec(h2o_p95))))
shared_clims = (0.0, vmax)
println("Shared color scale: 0 – ", round(vmax, digits=3), " wt%")

# =============================================================================
# PANELS
# =============================================================================
function h2o_panel(field, title_str; showbar=true)
    heatmap(
        t_C, p_vals, field,
        c              = :blues,
        clims          = shared_clims,     # <-- shared across all panels
        xlabel         = "Temperature (°C)",
        ylabel         = "Pressure (GPa)",
        title          = title_str,
        colorbar       = showbar,
        colorbar_title = showbar ? "wt%" : "",
    )
end

panels = [
    h2o_panel(h2o_p05, "Bound H₂O — p05 (driest)"),
    h2o_panel(h2o_p50, "Bound H₂O — p50 (median)"),
    h2o_panel(h2o_p95, "Bound H₂O — p95 (wettest)"),
]
ncols = 3

if SHOW_RANGE
    h2o_range = h2o_p95 .- h2o_p05
    push!(panels, heatmap(
        t_C, p_vals, h2o_range,
        c              = :heat,            # own scale — it's a different quantity
        xlabel         = "Temperature (°C)",
        ylabel         = "Pressure (GPa)",
        title          = "H₂O uncertainty  p95 − p05",
        colorbar_title = "wt%",
    ))
    ncols = 4
end

# =============================================================================
# COMBINE AND SAVE
# =============================================================================
fig = plot(panels...,
    layout = (1, ncols),
    size   = (520 * ncols, 460),
    margin = 5Plots.mm,
    plot_title = "Ensemble bound H₂O percentiles — $(SCENARIO)",
)

mkpath(dirname(OUTPATH))
savefig(fig, OUTPATH)
println("\nSaved: ", OUTPATH)

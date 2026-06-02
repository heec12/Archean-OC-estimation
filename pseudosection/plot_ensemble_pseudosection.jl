"""
plot_ensemble_pseudosection.jl
==============================
Ensemble-level pseudosection diagnostic figure for the homogeneous crust scenario.

Panel layout (2 × 2):
  p1 (top-left)    : Dominant hydrous phase  — from a representative Perple_X
                     scratch dir (e.g. the sample closest to the p50 bulk comp)
  p2 (top-right)   : Full phase assemblage   — same scratch dir as p1
  p3 (bottom-left) : Bound H₂O vol%          — p50 ensemble CSV
  p4 (bottom-right): H₂O uncertainty range   — (p95 − p05) ensemble CSVs

Compared with scratch.jl this script:
  • Replaces the single-sample 1σ uncertainty panel (p4) with the p95−p05
    interquartile range derived from the posterior ensemble.
  • Uses p50 (median) as the representative field for panels p3; panels p1/p2
    are still from one Perple_X run (the p50-nearest sample — see REP_SCRATCHDIR).
  • Reads the v2 CR-only CSV format produced by the Julia Perple_X pipeline.

Usage
-----
  julia plot_ensemble_pseudosection.jl

Edit the PATHS block below to point at your files.
"""

using StatGeochem, CSV, DataFrames, Statistics, LinearAlgebra, Plots

# =============================================================================
# PATHS  — edit these
# =============================================================================

# v2 ensemble CSV files (CR-only format, produced by the Perple_X Julia loop)
const H2O_LOOKUP_DIR = "/Users/hchoi342/Documents/Archean-OC/h2o_lookup"
# const SCENARIO       = "homogeneous_crust"
const SCENARIO       = "layered_cumulate_lower_crust"

# Bayesian posterior ensemble (one row per Perple_X sample, 7 oxide columns).
# Used to identify the sample whose bulk composition is closest to the p50
# posterior mean — that sample's scratch dir is then used for panels p1/p2.
const ENSEMBLE_CSV   = "/Users/hchoi342/Documents/Archean-OC/Bayesian/bayesian_lower_crust_outputs/ensemble_$(SCENARIO).csv"

# Root directory where Perple_X scratch dirs live.
# Each run is expected at:  <PERPLEX_SCRATCH_ROOT>/homogeneous_crust_sample_<N>
const PERPLEX_SCRATCH_ROOT = "/tmp/perplex_ensemble"

p05_path  = joinpath(H2O_LOOKUP_DIR, "v2_h2o_p05_$(SCENARIO).csv")
p50_path  = joinpath(H2O_LOOKUP_DIR, "v2_h2o_p50_$(SCENARIO).csv")
p95_path  = joinpath(H2O_LOOKUP_DIR, "v2_h2o_p95_$(SCENARIO).csv")

# Output figure path
const OUTPATH = "/Users/hchoi342/Documents/Archean-OC/pseudosection/pseudosection_ensemble_$(SCENARIO).png"

# =============================================================================
# P-T GRID  (must match what was used during the Perple_X runs)
# =============================================================================
const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=40))  # bar
const T_VEC = collect(range(273.0, 1600.0, length=40))                 # Kelvin

const n_P = length(P_VEC)
const n_T = length(T_VEC)

T_C   = T_VEC .- 273.15   # for axis labels
P_GPa = P_VEC ./ 10000

# =============================================================================
# HELPER: read CR-only CSV (vanKeken / Perple_X pipeline format)
# =============================================================================
function read_cr_csv(path)
    raw    = read(path, String)
    rows   = split(raw, '\r')
    # Drop any trailing empty rows
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
# FIND p50-NEAREST ENSEMBLE SAMPLE
# =============================================================================
# Strategy:
#   1. Read the Bayesian posterior ensemble (200 × 7 bulk compositions).
#   2. Compute the oxide-wise median across all samples → p50 bulk vector.
#   3. Find the row (1-indexed) with the smallest Euclidean distance to p50.
#   4. Point REP_SCRATCHDIR at that sample's Perple_X scratch directory.
#
# The oxide columns in the ensemble CSV must match OXIDES order:
#   SiO2_pct, TiO2_pct, Al2O3_pct, FeOtot_pct, MgO_pct, CaO_pct, Na2O_pct

println("="^60)
println("Finding p50-nearest ensemble sample …")

ens_df      = CSV.read(ENSEMBLE_CSV, DataFrame)
ens_matrix  = Matrix{Float64}(ens_df)          # n_samples × 7

p50_bulk    = median(ens_matrix, dims=1)[1, :] # 7-element vector

dists       = [norm(ens_matrix[i, :] .- p50_bulk) for i in 1:size(ens_matrix, 1)]
rep_idx     = argmin(dists)                    # 1-based sample index

REP_SCRATCHDIR = joinpath(PERPLEX_SCRATCH_ROOT, "$(SCENARIO)_sample_$(rep_idx)")

println("  Ensemble size   : ", size(ens_matrix, 1), " samples")
println("  p50 bulk (wt%)  : ", round.(p50_bulk, digits=3))
println("  Nearest sample  : #$(rep_idx)  (dist = $(round(dists[rep_idx], digits=4)))")
println("  Scratch dir     : ", REP_SCRATCHDIR)
println("="^60)

# =============================================================================
# PHASE GRID  (panels p1 / p2) — built from representative Perple_X scratch dir
# =============================================================================
println("Building phase grid from: ", REP_SCRATCHDIR)

test        = perplex_query_modes(REP_SCRATCHDIR, [P_VEC[1]], [T_VEC[1]])
phase_names = [k for k in keys(test) if k ∉ ["T(K)", "P(bar)", "elements"]]
println("Phases found: ", phase_names)

phase_grids = Dict(p => fill(NaN, n_P, n_T) for p in phase_names)

for (j, P) in enumerate(P_VEC)
    result = perplex_query_modes(REP_SCRATCHDIR, fill(P, n_T), T_VEC)
    for phase in phase_names
        if haskey(result, phase)
            phase_grids[phase][j, :] = result[phase]
        end
    end
end
println("Phase grid built: ", n_P, " × ", n_T)

# =============================================================================
# PANEL p1: Dominant hydrous phase
# =============================================================================
hydrous_phases = ["Chl(W)", "law", "zo", "cz", "phA", "liz", "br"]

dominant = fill("", n_P, n_T)
for i in 1:n_P, j in 1:n_T
    best_phase = ""
    best_vol   = 0.0
    for phase in hydrous_phases
        if haskey(phase_grids, phase)
            v = phase_grids[phase][i, j]
            if !isnan(v) && v > best_vol
                best_vol   = v
                best_phase = phase
            end
        end
    end
    dominant[i, j] = best_phase
end

all_hydrous_active = unique(filter(!isempty, vec(dominant)))
println("Active hydrous phases: ", all_hydrous_active)

code     = Dict(p => i for (i, p) in enumerate(all_hydrous_active))
code[""] = 0
dom_float = [code[dominant[i,j]] / max(length(all_hydrous_active), 1)
             for i in 1:n_P, j in 1:n_T]

p1 = heatmap(
    T_C, P_GPa, dom_float,
    c        = :tab10,
    xlabel   = "Temperature (°C)",
    ylabel   = "Pressure (GPa)",
    title    = "Dominant hydrous phase (sample #$(rep_idx), p50-nearest)",
    colorbar = false,
    clims    = (0.0, 1.0),
)
# Annotate each hydrous-phase field at its centroid
for phase in all_hydrous_active
    mask  = dominant .== phase
    idxs  = findall(mask)
    isempty(idxs) && continue
    mean_i = mean(getindex.(idxs, 1))
    mean_j = mean(getindex.(idxs, 2))
    ti = T_C[round(Int, clamp(mean_j, 1, n_T))]
    pi = P_GPa[round(Int, clamp(mean_i, 1, n_P))]
    annotate!(p1, ti, pi, text(phase, 6, :white, :center))
end

# =============================================================================
# PANEL p2: Full phase assemblage
# =============================================================================
threshold = 2.0   # vol% — phases below this are considered absent

tracked = ["Chl(W)", "law", "zo", "cz", "Gt(HGP)", "Cpx(HGP)", "Opx(HGP)",
           "O(HGP)", "Fsp(HGP)", "phA", "liz", "br", "sph", "ru", "q", "ta"]

assemblage = fill("", n_P, n_T)
for i in 1:n_P, j in 1:n_T
    phases_present = String[]
    for phase in tracked
        if haskey(phase_grids, phase)
            v = phase_grids[phase][i, j]
            if !isnan(v) && v > threshold
                push!(phases_present, phase)
            end
        end
    end
    assemblage[i, j] = join(sort(phases_present), "+")
end

unique_assemblages = unique(filter(!isempty, vec(assemblage)))
println("Unique assemblages ($(length(unique_assemblages))):")
for (k, a) in enumerate(unique_assemblages)
    println("  $k: $a")
end

assem_code      = Dict(a => i for (i, a) in enumerate(unique_assemblages))
assem_code[""]  = 0
assem_float     = [assem_code[assemblage[i,j]] / max(length(unique_assemblages), 1)
                   for i in 1:n_P, j in 1:n_T]

p2 = heatmap(
    T_C, P_GPa, assem_float,
    c        = :tab20,
    xlabel   = "Temperature (°C)",
    ylabel   = "Pressure (GPa)",
    title    = "Phase assemblage (sample #$(rep_idx), p50-nearest)",
    colorbar = false,
    clims    = (0.0, 1.0),
)
for assem in unique_assemblages
    mask  = assemblage .== assem
    idxs  = findall(mask)
    isempty(idxs) && continue
    mean_i = mean(getindex.(idxs, 1))
    mean_j = mean(getindex.(idxs, 2))
    ti = T_C[round(Int, clamp(mean_j, 1, n_T))]
    pi = P_GPa[round(Int, clamp(mean_i, 1, n_P))]
    label = replace(assem, "+" => "+\n")
    annotate!(p2, ti, pi, text(label, 5, :white, :center))
end

# =============================================================================
# LOAD ENSEMBLE CSVs
# =============================================================================
println("\nLoading ensemble CSVs …")
p_ens, t_ens, h2o_p05 = read_cr_csv(p05_path)
_,     _,     h2o_p50 = read_cr_csv(p50_path)
_,     _,     h2o_p95 = read_cr_csv(p95_path)

t_ens_C = t_ens .- 273.15   # K → °C for axis labels

println("p50 H₂O range : ",
    round(minimum(filter(!isnan, h2o_p50)), digits=3), " – ",
    round(maximum(filter(!isnan, h2o_p50)), digits=3), " wt%")

h2o_range = h2o_p95 .- h2o_p05   # p95 − p05 uncertainty band
println("p95−p05 range : ",
    round(minimum(filter(!isnan, h2o_range)), digits=3), " – ",
    round(maximum(filter(!isnan, h2o_range)), digits=3), " wt%")

# =============================================================================
# PANEL p3: Median (p50) bound H₂O field
# =============================================================================
p3 = heatmap(
    t_ens_C, p_ens, h2o_p50,
    c              = :blues,
    xlabel         = "Temperature (°C)",
    ylabel         = "Pressure (GPa)",
    title          = "Bound H₂O wt% — p50 (median)",
    colorbar_title = "wt%",
)

# =============================================================================
# PANEL p4: H₂O uncertainty — p95 minus p05 range
# =============================================================================
p4 = heatmap(
    t_ens_C, p_ens, h2o_range,
    c              = :heat,
    xlabel         = "Temperature (°C)",
    ylabel         = "Pressure (GPa)",
    title          = "H₂O uncertainty  p95 − p05 (ensemble)",
    colorbar_title = "wt%",
)

# =============================================================================
# COMBINE AND SAVE
# =============================================================================
fig = plot(p1, p2, p3, p4,
    layout = (2, 2),
    size   = (1400, 1000),
    margin = 5Plots.mm,
)

mkpath(dirname(OUTPATH))
savefig(fig, OUTPATH)
println("\nSaved: ", OUTPATH)

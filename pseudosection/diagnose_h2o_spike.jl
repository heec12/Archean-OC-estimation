"""
diagnose_h2o_spike.jl
=====================
Diagnose the vertical bound-H₂O "spike" seen in the ensemble percentile plots.

Strategy: a real dehydration boundary slopes in P-T; a numerical artifact pins
to a fixed-T grid column. So we query werami phase modes for the representative
(p50-nearest) sample along a PRESSURE column at the spike temperature, and at
the two neighboring T columns. If a hydrous phase (law / Chl(W) / ta ...) is
present only in the spike column and absent in the neighbors, the spike is a
metastable-retention artifact, not real phase equilibria.

Usage
-----
  julia --project diagnose_h2o_spike.jl
Edit T_SPIKE_C to the temperature of the spike you see in the figure (~600 °C).
"""

using StatGeochem, CSV, DataFrames, Statistics, LinearAlgebra

# =============================================================================
# PATHS / SETTINGS — match plot_ensemble_pseudosection.jl
# =============================================================================
const SCENARIO            = "homogeneous_crust"
const ENSEMBLE_CSV        = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/Bayesian/bayesian_lower_crust_outputs/ensemble_$(SCENARIO).csv"
const PERPLEX_SCRATCH_ROOT = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/perplex_ensemble"
const PERPLEX_VERSION       = "v1"

const T_SPIKE_C = 600.0    # °C — temperature of the vertical spike in the figure

# Same P-T grid as the Perple_X runs
const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=40))  # bar
const T_VEC = collect(range(273.0, 1600.0, length=40))                # K
const n_P, n_T = length(P_VEC), length(T_VEC)

const HYDROUS = ["law", "Chl(W)", "ta", "zo", "cz", "phA", "liz", "br"]

# =============================================================================
# Find p50-nearest sample (same logic as the plotting script)
# =============================================================================
ens     = Matrix{Float64}(CSV.read(ENSEMBLE_CSV, DataFrame))
p50     = median(ens, dims=1)[1, :]
rep_idx = argmin([norm(ens[i, :] .- p50) for i in 1:size(ens, 1)])
scratch = joinpath(PERPLEX_SCRATCH_ROOT, "$(PERPLEX_VERSION)_$(SCENARIO)_sample_$(rep_idx)")
println("Representative sample: #$rep_idx")
println("Scratch dir: $scratch\n")

# =============================================================================
# Locate the spike T column and its two neighbors
# =============================================================================
T_spike_K = T_SPIKE_C + 273.15
j_spike   = argmin(abs.(T_VEC .- T_spike_K))
j_cols    = filter(j -> 1 <= j <= n_T, [j_spike - 1, j_spike, j_spike + 1])
println("Spike column: j=$j_spike  (T=$(round(T_VEC[j_spike]-273.15,digits=1)) °C)")
println("Comparing columns: ", [round(T_VEC[j]-273.15, digits=1) for j in j_cols], " °C\n")

# =============================================================================
# Query modes along each T column (vary P) and report hydrous phases
# =============================================================================
for j in j_cols
    Tj   = T_VEC[j]
    res  = perplex_query_modes(scratch, P_VEC, fill(Tj, n_P))
    println("─"^64)
    println("T = $(round(Tj-273.15, digits=1)) °C", j == j_spike ? "   <-- SPIKE COLUMN" : "")
    println("  P (GPa) | ", join(rpad.(HYDROUS, 8), ""))
    for i in 1:n_P
        vals = [haskey(res, ph) && !isnan(res[ph][i]) ? res[ph][i] : 0.0 for ph in HYDROUS]
        sum(vals) < 0.01 && continue   # skip fully anhydrous nodes
        P_gpa = P_VEC[i] / 10000
        println("  ", rpad(round(P_gpa, digits=2), 8), "| ",
                join(rpad.(round.(vals, digits=2), 8), ""))
    end
end

println("\n" * "─"^64)
println("Read: if a hydrous phase (esp. law / Chl(W)) carries vol% in the SPIKE")
println("column at high P but is ~0 in BOTH neighbor columns, the spike is a")
println("metastable-retention artifact. Confirm by re-running this scenario at")
println("80×80 — a real field persists, an artifact narrows or disappears.")

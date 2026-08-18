"""
diagnose_h2o_spike_members.jl
=============================
The percentile bound-H₂O panels aggregate all 200 ensemble members per pixel,
so a vertical spike there can come from a SUBSET of members even when the
representative member (#44) is clean. This script tests the "local-max at the
front column" hypothesis directly: for each member it queries the spike T
column and its two T-neighbours, and counts pressure nodes where the spike
column is a strict local maximum in hydrous content (wetter than BOTH
neighbours). Members with many such nodes are the ones painting the stripe.

A clean, monotonic-dehydration member (like #44) scores ~0. A member that
metastably retains talc/lawsonite one column late at the front scores high.

Usage
-----
  julia --project diagnose_h2o_spike_members.jl
"""

using StatGeochem, CSV, DataFrames, Statistics

const SCENARIO             = "homogeneous_crust"
const PERPLEX_VERSION      = "v1"
const PERPLEX_SCRATCH_ROOT = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/perplex_ensemble"
const ENSEMBLE_CSV        = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/Bayesian/bayesian_lower_crust_outputs/ensemble_$(SCENARIO).csv"

const T_SPIKE_C   = 612.3   # °C — spike column
const MAX_SAMPLES = 40      # raise to 200 for the full ensemble (slower)
const P_MIN_GPA   = 4.0     # only count local maxima above this P (the stripe lives high-P)
const EPS_VOL     = 0.5     # vol% margin to call a strict local max

const HYDROUS = ["law", "Chl(W)", "ta", "zo", "cz", "phA", "liz", "br"]

# Same P-T grid as the runs
const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=40))  # bar
const T_VEC = collect(range(273.0, 1600.0, length=40))                # K
const n_P   = length(P_VEC)

# Spike column + neighbours
j_spk = argmin(abs.((T_VEC .- 273.15) .- T_SPIKE_C))
T_lo, T_mid, T_hi = T_VEC[j_spk-1], T_VEC[j_spk], T_VEC[j_spk+1]
println("Spike column T=$(round(T_mid-273.15,digits=1)) °C; neighbours ",
        round.([T_lo, T_hi] .- 273.15, digits=1), " °C")
println("Counting local maxima above $(P_MIN_GPA) GPa\n")

# Sum hydrous vol% over the standard phase list, treating NaN/missing as 0
hydsum(res, k) = sum(haskey(res, ph) && !isnan(res[ph][k]) ? max(0.0, res[ph][k]) : 0.0
                     for ph in HYDROUS)

n_ens = size(CSV.read(ENSEMBLE_CSV, DataFrame), 1)
scan  = 1:min(MAX_SAMPLES, n_ens)

# One werami call per sample: the three columns stacked
Pall = vcat(P_VEC, P_VEC, P_VEC)
Tall = vcat(fill(T_lo, n_P), fill(T_mid, n_P), fill(T_hi, n_P))

results = Tuple{Int,Int,Float64}[]   # (sample, localmax_count, max_excess_vol%)
for s in scan
    scratch = joinpath(PERPLEX_SCRATCH_ROOT,
                       "$(PERPLEX_VERSION)_$(SCENARIO)_sample_$(s)")
    isdir(joinpath(scratch, "out1")) || (println("  #$s : no build, skip"); continue)
    try
        res = perplex_query_modes(scratch, Pall, Tall)
        cnt, mx = 0, 0.0
        for i in 1:n_P
            (P_VEC[i] / 10000) < P_MIN_GPA && continue
            lo  = hydsum(res, i)            # 578 chunk
            mid = hydsum(res, i + n_P)      # 612 chunk
            hi  = hydsum(res, i + 2n_P)     # 646 chunk
            excess = mid - max(lo, hi)
            if excess > EPS_VOL
                cnt += 1
                mx = max(mx, excess)
            end
        end
        push!(results, (s, cnt, mx))
    catch e
        println("  #$s : query failed ($e)")
    end
end

sort!(results, by = r -> -r[2])
println("\nMembers ranked by spike-column local-max count (high = paints the stripe):")
println("  sample | local-max nodes | max excess (vol%)")
for (s, cnt, mx) in results
    flag = cnt > 0 ? (s == 44 ? "" : "  <-- contributor") : ""
    println("  ", rpad(s, 7), "| ", rpad(cnt, 16), "| ", rpad(round(mx, digits=2), 8), flag)
end

contrib = count(r -> r[2] > 0, results)
println("\n$(contrib) / $(length(results)) scanned members show a local-max at the spike column.")
println("If many do, the percentile stripe is metastable retention at the steep")
println("dehydration front (fix on the Perple_X build side: tighter auto-refine /")
println("denser nodes, or an 80×80 rerun). If ~none do, the stripe is introduced in")
println("the percentile-aggregation step — check NaN handling and grid alignment there.")

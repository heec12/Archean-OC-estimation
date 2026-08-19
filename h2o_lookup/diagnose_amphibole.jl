# diagnose_amphibole.jl
# =====================
# Why this exists
# ---------------
# In the p0091_lower smoke test, cAmph(G) did not appear at 0.5/2.0/4.0 GPa.
# In an ultramafic bulk with CaO 5.8 wt% that MIGHT be genuine -- brucite-
# bearing assemblages have low silica activity, which disfavours tremolite,
# and diopside takes the Ca instead. But it might also mean the EXCLUDES list
# is crippling the model, or that cAmph(G) is being silently rejected the way
# Sp(HGP) and Amph(DPW) were.
#
# The distinction matters a lot: amphibole dominates the bound-H2O budget in
# the UPPER crust runs. If it cannot form, roughly half the design returns a
# spuriously dry answer and it will look like a real compositional signal.
#
# What this does
# --------------
# Runs three real compositions from composition_manifest.csv across the P-T
# window where Ca-amphibole should be most stable (0.5-2.0 GPa, 650-900 K),
# twice: once with the production EXCLUDES list and once with excludes
# switched off. Prints a compact table of amphibole wt% and bound H2O rather
# than dumping full point output.
#
# Reading the result
# ------------------
#   amphibole present WITH excludes            -> production config is fine
#   absent with excludes, present without      -> EXCLUDES is the problem;
#                                                 ts/parg/gl/ged/fanth are
#                                                 endmembers of cAmph(G) and
#                                                 removing them guts the model
#   absent in BOTH, for the basaltic composition
#                                              -> cAmph(G) is being rejected;
#                                                 check the build log
#   absent only for the ultramafic composition -> genuine petrology, no action
#
# The basaltic case (p0014_upper, CaO 15.5, Al2O3 14.7) is the diagnostic one.
# If amphibole is missing THERE, something is broken.
#
# Usage
# -----
#   julia --project=. diagnose_amphibole.jl
#
# Runs 6 pseudosections; expect roughly 10-20 minutes. Compute node, not login.

using StatGeochem
using StatGeochem.Perple_X_jll
using Printf

const BASE = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation"
const SCRATCH_ROOT = joinpath(BASE, "perplex_amph_diag")

const COMPONENTS = ["SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "H2O"]
const H2O_EXCESS = 20.0

# Production solution model list, unchanged from run_perplex_manifest.jl
const SOLUTIONS = "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nFsp(HGP)\nSp(WPC)\n" *
                  "Chl(W)\ncAmph(G)\nAtg(PN)\nT\nB\n"

# ts   = tschermakite     ) all four are cAmph(G) endmembers
# parg = pargasite        )
# gl   = glaucophane      )
# ged  = gedrite          ) ged and fanth are ORTHOamphibole endmembers, so
# fanth= ferroanthophyllite) excluding them is harmless for cAmph(G) but would
#                           cripple oAmph(DP) if that were ever added
const EXCLUDES_PROD = "ts\nparg\ngl\nged\nfanth\n"
const EXCLUDES_NONE = ""

# Real compositions from composition_manifest.csv (anhydrous, wt%)
const CASES = [
    ("p0014_upper_basaltic",
     [49.3060, 0.7531, 14.7055, 10.8415,  7.8565, 15.5106, 1.0269],
     "basaltic upper crust, CaO 15.5 / Al2O3 14.7 -- amphibole SHOULD form"),
    ("p0082_upper_highMgO",
     [49.0338, 0.9360,  6.5308, 15.5044, 17.5501, 10.3599, 0.0851],
     "high-MgO upper crust, the hottest end of the upper-crust design"),
    ("p0091_lower_ultramafic",
     [46.1496, 0.1798,  3.1510,  8.8598, 35.7060,  5.8234, 0.1305],
     "ultramafic lower crust -- the composition from the smoke test"),
]

# Amphibole field: 0.5-2.0 GPa, 650-900 K
const P_POINTS = [5000.0, 10000.0, 15000.0, 20000.0]      # bar
const T_POINTS = [650.0, 750.0, 850.0, 900.0]             # K

# ---------------------------------------------------------------------------

"""
Extract phase name => wt% from a point query, plus the Solid Only H2O wt%.
Same block logic and column indices as run_perplex_manifest.jl, verified
against real smoke-test output.
"""
function parse_phases(point_str::String)
    phases = Dict{String,Float64}()
    h2o_solid = NaN
    in_phase = false
    in_bulk = false
    has_solid_only = false

    for line in split(point_str, "\n")
        if occursin("Phase Compositions (weight percentages)", line)
            in_phase = true; continue
        end
        if in_phase && (occursin("Phase speciation", line) ||
                        occursin("Bulk Composition", line))
            in_phase = false
        end
        if in_phase
            t = split(strip(line))
            if length(t) == 13
                wt = tryparse(Float64, t[2])
                wt === nothing || (phases[t[1]] = wt)
            end
        end

        if occursin("Bulk Composition", line); in_bulk = true; continue; end
        if in_bulk
            occursin("Solid Only", line) && (has_solid_only = true)
            t = split(strip(line))
            if length(t) >= 4 && t[1] == "H2O"
                h2o_solid = has_solid_only && length(t) >= 8 ?
                    something(tryparse(Float64, t[8]), NaN) :
                    something(tryparse(Float64, t[4]), NaN)
            end
        end
    end
    return phases, h2o_solid
end

is_amphibole(name) = occursin("Amph", name) || name in ("tr", "cumm", "anth", "act")

function run_case(tag, comp7, excludes, label)
    scratch = joinpath(SCRATCH_ROOT, tag)
    rm(scratch, recursive = true, force = true); mkpath(scratch)

    comp = vcat(comp7, H2O_EXCESS)

    println("\n" * "="^74)
    println("  $tag   [excludes: $(isempty(excludes) ? "NONE" : "production")]")
    println("="^74)
    flush(stdout)

    try
        perplex_configure_pseudosection(
            scratch, comp, COMPONENTS,
            (minimum(P_POINTS), maximum(P_POINTS)),
            (minimum(T_POINTS), maximum(T_POINTS)),
            dataset         = "hp62ver.dat",
            solution_phases = SOLUTIONS,
            excludes        = excludes,
            fluid_eos       = 5,
        )
    catch e
        @warn "configure failed for $tag" exception = e
        return
    end

    @printf("%-9s %-7s  %9s  %9s   %s\n",
            "P(GPa)", "T(K)", "amph wt%", "H2O sol", "assemblage")
    println("-"^74)

    amph_seen = false
    for P in P_POINTS, T in T_POINTS
        phases, h2o = try
            parse_phases(perplex_query_point(scratch, P, T))
        catch e
            (Dict{String,Float64}(), NaN)
        end
        amph = sum(v for (k, v) in phases if is_amphibole(k); init = 0.0)
        amph > 0.0 && (amph_seen = true)
        names = join(sort(collect(keys(phases))), " ")
        @printf("%-9.2f %-7.0f  %9.2f  %9.2f   %s\n",
                P / 10000, T, amph, h2o, names)
        flush(stdout)
    end

    println("-"^74)
    println(amph_seen ? "  AMPHIBOLE PRESENT somewhere in this window" :
                        "  NO AMPHIBOLE anywhere in this window")
    println("  ($label)")

    # Surface silent model rejections -- the failure mode that does not
    # announce itself.
    for f in filter(x -> endswith(x, ".log") || endswith(x, ".txt"),
                    readdir(scratch, join = true))
        for line in eachline(f)
            if occursin(r"(?i)reject|not found|invalid|missing endmember"i, line)
                println("  BUILD NOTE [", basename(f), "]: ", strip(line))
            end
        end
    end

    rm(scratch, recursive = true, force = true)
end

# ---------------------------------------------------------------------------

mkpath(SCRATCH_ROOT)
println("Amphibole stability diagnostic")
println("  solution models: ", replace(strip(SOLUTIONS), "\n" => ", "))
println("  H2O excess     : $H2O_EXCESS wt%")
println("  P range        : $(minimum(P_POINTS)/10000)-$(maximum(P_POINTS)/10000) GPa")
println("  T range        : $(minimum(T_POINTS))-$(maximum(T_POINTS)) K")
flush(stdout)

for (tag, comp, label) in CASES
    run_case(tag * "_excl", comp, EXCLUDES_PROD, label)
    run_case(tag * "_noexcl", comp, EXCLUDES_NONE, label)
end

rm(SCRATCH_ROOT, recursive = true, force = true)

println("\n" * "="^74)
println("Interpretation")
println("="^74)
println("""
  Compare each composition's two blocks.

  * Amphibole present WITH production excludes
        -> nothing to fix, the smoke test result was genuine petrology.

  * Absent with excludes, present without
        -> EXCLUDES is suppressing cAmph(G). ts, parg and gl are its
           endmembers. Drop them from EXCLUDES in run_perplex_manifest.jl
           and rerun. Keep ged and fanth, which belong to orthoamphibole.

  * Absent in BOTH for p0014_upper_basaltic
        -> cAmph(G) is not loading at all. Check the build log for a
           rejection and verify the model name against solution_model.dat.
           This would invalidate every upper-crust run in the design.

  * Absent only for p0091_lower_ultramafic
        -> expected. Low silica activity in brucite-bearing assemblages
           disfavours tremolite; diopside takes the Ca. No action.
""")

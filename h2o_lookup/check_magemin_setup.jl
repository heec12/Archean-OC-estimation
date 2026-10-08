# check_magemin_setup.jl
# ======================
# Run once, interactively, before submitting any v4 job:
#
#     julia --project=. h2o_lookup/check_magemin_setup.jl
#
# It checks three things the v4 driver depends on:
#   1. MAGEMin_C loads and the three databases (mb, ume, ig) initialise.
#   2. The phases the design relies on exist in this MAGEMin version:
#        ume -> atg, br, chl, ta (serpentine-group carriers) and DEW (to remove)
#        ig  -> liq, amp, chl     (and NO atg / br: that is expected)
#        mb  -> liq, amp, law, mu, dio
#   3. Three reference points reproduce the values computed while building v4
#      (MAGEMin 2.0.7 command-line build, Oct 2026). Agreement to ~0.05 wt%
#      bound H2O and the same assemblage means the driver's bookkeeping and
#      your MAGEMin install agree with the pilot numbers in the notes.

using MAGEMin_C

const OX = ["SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3", "MgO", "CaO", "Na2O", "K2O", "H2O"]

# Compositions straight from composition_manifest_v4.csv (wt%), + 20 wt% H2O.
const REF = [
    (run = "p0091_lower_ume_kmorb", db = "ume", P_GPa = 1.682747, T_K = 534.930442,
     X = nothing, h2o = 9.957, asm = "atg+aug+br+chl+fl+spi"),
    (run = "p0091_lower_ig_kmorb",  db = "ig",  P_GPa = 1.682747, T_K = 534.930442,
     X = nothing, h2o = 10.012, asm = "bi+chl+cpx+fl+g+ol"),
    (run = "p0040_upper_mb_kmorb",  db = "mb",  P_GPa = 1.46169,  T_K = 517.151823,
     X = nothing, h2o = 6.855, asm = "H2O+amp+amp+chl+law+mu+q+sph"),
]

const FLUID = Set(["H2O", "fl", "flc", "DEW", "aq17"])
const MELT  = Set(["liq"])
const SETTINGS = Dict("mb"  => (mbCpx = 0, remove = String[]),
                      "ume" => (mbCpx = 1, remove = ["DEW", "flc", "occm", "po"]),
                      "ig"  => (mbCpx = 1, remove = String[]))

println("MAGEMin_C ", pkgversion(MAGEMin_C))

# ---- 2. phase inventories ---------------------------------------------------
need = Dict("ume" => ["atg", "br", "chl", "ta"], "ig" => ["liq", "amp", "chl"],
            "mb" => ["liq", "amp", "mu", "dio"])
for db in ("mb", "ume", "ig")
    info = retrieve_solution_phase_information(db)
    ss = info.ss_name
    miss = setdiff(need[db], ss)
    println(rpad(db, 4), " solution phases: ", join(ss, " "))
    println(rpad("", 4), " pure phases    : ", join(info.data_pp, " "))
    println(rpad("", 4), isempty(miss) ? " OK: required phases present" :
                                         " !! MISSING: " * join(miss, ", "))
end
println("ig has atg/br? ", any(in(["atg", "br"]), retrieve_solution_phase_information("ig").ss_name),
        "  (expected false)")

# ---- 3. reference points ----------------------------------------------------
using CSV, DataFrames
manifest = CSV.read(joinpath(@__DIR__, "..", "design_outputs", "composition_manifest_v4.csv"), DataFrame)

for r in REF
    row = manifest[findfirst(==(r.run), manifest.run_id), :]
    X = vcat(Float64[row[o] for o in OX[1:end-1]], 20.0)
    s = SETTINGS[r.db]
    data = Initialize_MAGEMin(r.db, verbose = false, mbCpx = s.mbCpx)
    present = retrieve_solution_phase_information(r.db).ss_name
    rmv = filter(in(present), s.remove)
    rm_list = isempty(rmv) ? nothing : remove_phases(rmv, r.db)
    out = multi_point_minimization([10 * r.P_GPa], [r.T_K - 273.15], data;
                                   X = X, Xoxides = OX, sys_in = "wt",
                                   rm_list = rm_list, progressbar = false)[1]
    iH = findfirst(==("H2O"), out.oxides)
    ms = 0.0; ws = 0.0
    for k in 1:(out.n_SS + out.n_PP)
        nm = out.ph[k]
        (nm in FLUID || nm in MELT) && continue
        c = k <= out.n_SS ? out.SS_vec[k].Comp_wt : out.PP_vec[k - out.n_SS].Comp_wt
        ms += out.ph_frac_wt[k]; ws += out.ph_frac_wt[k] * c[iH]
    end
    h = 100 * ws / ms
    asm = join(sort(out.ph[1:(out.n_SS + out.n_PP)]), "+")
    ok = abs(h - r.h2o) < 0.05 && asm == r.asm
    println("\n", r.run, "  (", r.db, ", ", r.P_GPa, " GPa, ", round(r.T_K), " K)")
    println("  bound H2O : ", round(h, digits = 3), " wt%   reference ", r.h2o)
    println("  assemblage: ", asm)
    println("  reference : ", r.asm)
    println(ok ? "  OK" : "  !! DIFFERS - check MAGEMin version / phase removal before running v4")
    Finalize_MAGEMin(data)
end

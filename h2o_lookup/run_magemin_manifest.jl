# run_magemin_manifest.jl
# =======================
# Stage 2 of the v4 Archean oceanic crust bound-H2O pipeline.
# Replaces run_perplex_manifest.jl (v3, Perple_X / StatGeochem).
#
# WHAT CHANGED FROM v3 AND WHY
# ----------------------------
# * Engine: MAGEMin (MAGEMin_C.jl) instead of Perple_X. MAGEMin ships the
#   THERMOCALC a-X sets as self-consistent databases, so each lithology gets
#   one coherent set rather than a hand-assembled solution-model list
#   (Palin, pers. comm. Oct 2026: "the a-X relations are often presented as
#   sets ... I'd prefer to use self-consistent sets for each lithology").
#
# * Chemical system: KNCFMASHTO (K2O and Fe3+ added). Compositions come
#   from composition_manifest_v4.csv (augment_manifest_v4.py).
#
# * Database per manifest row (column `db`):
#     upper crust  -> "mb"   Green et al. 2016 metabasite set (with melt)
#     lower crust  -> "ume"  Evans & Frost 2021 + G16 pl/amp/aug  [primary]
#                     "ig"   Green et al. 2025                     [check]
#   `ume` has antigorite + brucite but NO melt and a Ca-free (py-alm) garnet.
#   `ig` has melt and full garnet but NO antigorite or brucite. Neither is
#   right everywhere, so both are run and post-processing stitches them
#   (ume up to antigorite-out, ig beyond). This script never chooses.
#
# * Geometry: calculations along the slab P-T paths (design_outputs/pt_paths),
#   plus the P-T grid for the TerraFERMA lookup table. The path points are what
#   the reaction-resolved dehydration metrics need; the grid is what TerraFERMA
#   reads.
#
# * Output is reaction-resolved. Two long-format tables per run:
#     pts_<run_id>.csv     one row per P-T point (bound H2O, fluid, melt, flags)
#     phases_<run_id>.csv  one row per stable phase per point, with that
#                          phase's H2O content and its contribution to the
#                          rock's bound H2O.
#   The second table is what lets post-processing say "chlorite-out released
#   X wt% at Y km" instead of quoting a 50% threshold.
#
# KEPT FROM v3
# ------------
# * No statistics here. Every reduction happens in post-processing.
# * H2O in excess (20 wt%) so bound H2O is a CAPACITY, and saturation is
#   checked, not assumed (cells with no free fluid are flagged).
# * Restartable (finished runs are skipped) and SLURM-array aware (rows split
#   round-robin across tasks).
#
# KNOWN LIMITS (also in Methodology Updates.md)
# ---------------------------------------------
# * With 20 wt% excess H2O, melt amounts above the wet solidus are
#   overstated. Points with melt are FLAGGED (melt_present) and must not be
#   used as bound-H2O capacities. The minimum-water melt calculation (Palin)
#   is a separate, later step.
# * `ume` has no TiO2 and no K2O component: those oxides are dropped and the
#   rest renormalised by MAGEMin. The dropped mass is reported per run.
# * Databases use different end-member datasets by default (mb: ds62,
#   ume: ds633, ig: ds636). Recorded in the output; matters when comparing
#   ume and ig directly.
#
# SETUP (once, interactively, at the repo root)
# ---------------------------------------------
#   julia --project=. -e 'using Pkg; Pkg.add("MAGEMin_C"); using MAGEMin_C'
#   (written against MAGEMin_C 2.0.x; uses only Initialize_MAGEMin,
#    multi_point_minimization, remove_phases, retrieve_solution_phase_information,
#    Finalize_MAGEMin, which have been stable across 1.x-2.x)
#
# USAGE
# -----
#   julia --project=. --threads 8 h2o_lookup/run_magemin_manifest.jl \
#         --manifest design_outputs/composition_manifest_v4.csv \
#         --paths design_outputs/pt_paths --outdir h2o_runs/v4
#
#   options:  --pilot true      only rows flagged pilot (6 pairs)
#             --scenario v4a    only rows used by one scenario (v4a | v4b | v4)
#             --db ume          only rows for one database
#             --grid false      paths only (fast; skip the lookup grid)
#             --rows 1:10       explicit manifest row range (1-based)
#             --version v4      tag written into output dir name (default v4)

using MAGEMin_C
using CSV, DataFrames, Printf

# =============================================================================
# SETTINGS
# =============================================================================

# Anhydrous oxides as written by augment_manifest_v4.py, wt%, sum 100.
const OXIDES = ["SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3",
                "MgO", "CaO", "Na2O", "K2O"]

# Excess H2O, wt% added on top of the 100 wt% anhydrous composition.
# Same reasoning as v3: fully hydrated serpentinite + brucite holds 12-15 wt%,
# so 20 wt% keeps free fluid present and makes bound H2O a capacity.
const H2O_EXCESS = 20.0

# Phase names treated as free fluid (excluded from the solid) and as melt.
#   mb: pure-phase "H2O"     ig: "fl" (+ pure "H2O")
#   ume: "fl" (H2-H2O fluid), pure "H2O"; "flc"/"DEW" are removed below but
#   listed here so that, if ever re-enabled, they are not counted as solids.
const FLUID_NAMES = Set(["H2O", "fl", "flc", "DEW", "aq17"])
const MELT_NAMES  = Set(["liq"])
# In the `all` database phases carry the model tag (e.g. "liq_G16", "fl_G25",
# "atg_EF21"). base_name() strips it so the same rules and the same
# post-processing apply to every database.
base_name(ph::AbstractString) = String(first(split(ph, "_")))
is_fluid(ph) = base_name(ph) in FLUID_NAMES
is_melt(ph)  = base_name(ph) in MELT_NAMES

# Below this free-fluid wt% (of the whole system) a point is undersaturated:
# its bound H2O is a censored value, not a capacity.
const FLUID_PRESENT_THRESHOLD = 0.01   # wt%
const MELT_PRESENT_THRESHOLD  = 0.01   # wt%

# Per-database options.
#   mbCpx = 0 -> omphacite ("dio") instead of augite. Slab cpx becomes
#                omphacitic with depth; augite (Green et al. 2016) is not meant
#                for Na-rich compositions. Revisit in the pilot (run mbCpx=1
#                on a few rows and compare) before the production run.
#   dataset    -> end-member dataset (nothing = the database default).
#   keep       -> if given, ONLY these solution phases are active (include
#                 list, used with the `all` database); otherwise `remove`.
#   remove     -> solution phases switched off.
#                ume: DEW (aqueous speciation model) is ON by default in ume.
#                     It dissolves MgO, SiO2, Na2O into the fluid and changes
#                     the solid assemblage; v3 used pure-H2O fluid, so it is
#                     switched off for comparability. flc/occm are the
#                     H2O-CO2 fluid and carbonate (CO2 = 0 here); po needs S.
#                     Names not present in the installed MAGEMin version are
#                     skipped with a note, not an error.
#
# Keys are `db` or `db:phase_set` from the manifest.
#
# "all:v3like" (scenario v4a) -- closest MAGEMin replica of the v3 Perple_X
# list, run on ds62 like v3 (hp62ver.dat):
#     v3 Perple_X     -> MAGEMin `all`
#     cAmph(G)        -> amp_G16    same model (v3 had ts/parg/gl excluded; this does not)
#     O(HGP)          -> ol_H18     same model
#     Cpx/Opx/Gt(HGP) -> cpx_T21, opx_T21, g_T21   corrected successors of HGP 2018
#                                  (g_H18 in MAGEMin is the 2-end-member EF21 garnet: NOT used)
#     Fsp(HGP)        -> fsp_H22    newer feldspar model
#     Sp(WPC)         -> spl_W02    same model family (no magnetite: Fe3+ = 0 in v4a)
#     Chl(W)          -> chl_W14    same model
#     Atg(PN), T, B   -> atg_EF21, ta_EF21, br_E13   different models (EF21 set)
#   Fluid: pure H2O phase (no fl_* model selected), as in v3.
#   Checked with the MAGEMin 2.0.7 CLI: on ds636 instead of ds62 the same list
#   loses amphibole in a basalt at 2 GPa / 550 C, so the dataset choice matters.
const V3LIKE = ["amp_G16", "ol_H18", "cpx_T21", "opx_T21", "g_T21",
                "fsp_H22", "spl_W02", "chl_W14", "atg_EF21", "ta_EF21", "br_E13"]

const DB_SETTINGS = Dict(
    "mb"         => (db = "mb",  mbCpx = 0, dataset = nothing, keep = nothing, remove = String[]),
    "ume"        => (db = "ume", mbCpx = 1, dataset = nothing, keep = nothing, remove = ["DEW", "flc", "occm", "po"]),
    "ig"         => (db = "ig",  mbCpx = 1, dataset = nothing, keep = nothing, remove = String[]),
    "all:v3like" => (db = "all", mbCpx = 1, dataset = 62,      keep = V3LIKE,  remove = String[]),
)

# Settings key for a manifest row: "db" or "db:phase_set".
function db_key(row)
    ps = hasproperty(row, :phase_set) ? row.phase_set : missing
    (ismissing(ps) || isempty(strip(string(ps))) || string(ps) == "default") ?
        String(row.db) : String(row.db) * ":" * string(ps)
end

# P-T grid for the TerraFERMA lookup table (same extent as v3).
# MAGEMin units: P in kbar, T in deg C.
const GRID_P_GPA = collect(range(0.1, 8.0, length = 40))
const GRID_T_K   = collect(range(273.0, 1600.0, length = 40))

# =============================================================================
# ARGUMENTS
# =============================================================================

function parse_args()
    opts = Dict{String,String}(
        "manifest" => joinpath("design_outputs", "composition_manifest_v4.csv"),
        "paths"    => joinpath("design_outputs", "pt_paths"),
        "outdir"   => "",
        "version"  => "v4",
        "grid"     => "true",
        "pilot"    => "false",
        "db"       => "",
        "scenario" => "",
        "rows"     => "",
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if startswith(a, "--") && i < length(ARGS)
            opts[a[3:end]] = ARGS[i+1]
            i += 2
        else
            i += 1
        end
    end
    return opts
end

istrue(s) = lowercase(s) in ("true", "1", "yes")

const OPTS       = parse_args()
const VERSION_TAG = OPTS["version"]
const OUTPUT_DIR = isempty(OPTS["outdir"]) ? joinpath("h2o_runs", VERSION_TAG) : OPTS["outdir"]
const DO_GRID    = istrue(OPTS["grid"])
mkpath(OUTPUT_DIR)

# =============================================================================
# P-T PATHS
# =============================================================================
"""
Reads every pt_path_<name>.csv in the paths directory. Expected columns:
depth_km, P_GPa, T_K (the TerraFERMA export used in v3). Returns a vector of
(name, depth_km, P_GPa, T_K).
"""
function load_paths(dir::String)
    paths = []
    isdir(dir) || (@warn "no P-T path directory at $dir"; return paths)
    for f in sort(readdir(dir))
        m = match(r"^pt_path_(.+)\.csv$", f)
        m === nothing && continue
        df = CSV.read(joinpath(dir, f), DataFrame)
        push!(paths, (name = String(m.captures[1]),
                      depth_km = Float64.(df.depth_km),
                      P_GPa = Float64.(df.P_GPa),
                      T_K = Float64.(df.T_K)))
    end
    return paths
end

# =============================================================================
# MAGEMin DATABASES (initialised lazily, one per database per process)
# =============================================================================

const DBS = Dict{String,Any}()
const RM_LISTS = Dict{String,Any}()

function magemin_db(key::String)
    haskey(DBS, key) && return DBS[key]
    s = DB_SETTINGS[key]
    db = s.db
    data = s.dataset === nothing ?
        Initialize_MAGEMin(db, verbose = false, mbCpx = s.mbCpx) :
        Initialize_MAGEMin(db, verbose = false, mbCpx = s.mbCpx, dataset = s.dataset)

    info = retrieve_solution_phase_information(db)
    if s.keep !== nothing
        # Include list. select_phases accepts the tagged names ("amp_G16").
        tagged = [x.ss_fName for x in info.data_ss]
        missing_ = filter(x -> !(x in tagged) && !(x in info.ss_name), s.keep)
        isempty(missing_) || error("[$key] phases not in this MAGEMin version: " *
                                   join(missing_, ", "))
        RM_LISTS[key] = select_phases(db; ss_list = s.keep)
        println("  [$key] initialised (dataset $(something(s.dataset, "default"))); ",
                "active solution phases: ", join(s.keep, ", "))
    else
        # Only remove phases that exist in this database / MAGEMin version.
        # (remove_phases errors on unknown names in some versions.)
        rm = filter(in(info.ss_name), s.remove)
        skipped = setdiff(s.remove, rm)
        isempty(skipped) || println("  [$key] not in this MAGEMin version, not removed: ",
                                    join(skipped, ", "))
        RM_LISTS[key] = isempty(rm) ? nothing : remove_phases(rm, db)
        println("  [$key] initialised; removed: ", isempty(rm) ? "none" : join(rm, ", "))
    end
    DBS[key] = data
    flush(stdout)
    return data
end

function oxide_list(db::String)
    try
        return MAGEMin_C.get_oxide_list(db)
    catch
        return String[]          # older versions: no dropped-mass report
    end
end

# =============================================================================
# ONE POINT -> rows
# =============================================================================
"""
Splits one MAGEMin result into solid / fluid / melt and per-phase H2O.

Mass bookkeeping (all from MAGEMin's own wt fractions):
    ph_frac_wt[i]          mass fraction of phase i in the whole system
    Comp_wt[iH2O]          mass fraction of H2O in phase i
Solid = every phase that is neither fluid nor melt. Bound H2O is reported
per unit mass of SOLID, i.e. the same quantity as Perple_X's "Solid Only"
column in v3, which is what TerraFERMA reads.
"""
function split_point(out)
    iH2O = findfirst(==("H2O"), out.oxides)
    n_SS, n_PP = out.n_SS, out.n_PP

    phases = NamedTuple[]
    m_solid = 0.0; w_solid = 0.0
    m_fluid = 0.0
    m_melt  = 0.0; w_melt = 0.0

    for k in 1:(n_SS + n_PP)
        name = out.ph[k]
        f = out.ph_frac_wt[k]
        comp = k <= n_SS ? out.SS_vec[k].Comp_wt : out.PP_vec[k - n_SS].Comp_wt
        xh2o = iH2O === nothing ? 0.0 : comp[iH2O]
        kind = is_fluid(name) ? "fluid" : is_melt(name) ? "melt" : "solid"
        if kind == "solid"
            m_solid += f; w_solid += f * xh2o
        elseif kind == "fluid"
            m_fluid += f
        else
            m_melt += f; w_melt += f * xh2o
        end
        push!(phases, (phase = base_name(name), model = name, kind = kind,
                        frac_sys = f, h2o_in_phase = xh2o))
    end

    h2o_solid = m_solid > 0 ? 100 * w_solid / m_solid : NaN
    return (h2o_solid = h2o_solid,
            solid_wt = 100 * m_solid,
            fluid_wt = 100 * m_fluid,
            melt_wt = 100 * m_melt,
            h2o_melt_sys = 100 * w_melt,
            m_solid = m_solid,
            phases = phases)
end

# =============================================================================
# ONE RUN
# =============================================================================

function composition(row)
    X = Float64[row[o] for o in OXIDES]
    return vcat(X, H2O_EXCESS), vcat(OXIDES, "H2O")
end

"""
Runs one manifest row: all P-T paths, then (optionally) the grid.
Writes pts_<run_id>.csv and phases_<run_id>.csv atomically.
"""
function run_one(row, paths)
    run_id = String(row.run_id)
    key = db_key(row)
    db = DB_SETTINGS[key].db
    f_pts = joinpath(OUTPUT_DIR, "pts_$(run_id).csv")
    f_ph  = joinpath(OUTPUT_DIR, "phases_$(run_id).csv")
    if isfile(f_pts) && isfile(f_ph)
        println("  [skip] $run_id"); flush(stdout)
        return :skipped
    end

    X, Xox = composition(row)
    ox_db = oxide_list(db)
    dropped = isempty(ox_db) ? String[] :
              [o for o in Xox if !(o in ox_db) && !(o in ("FeO", "Fe2O3"))]
    dropped_wt = sum((row[o] for o in dropped if o in OXIDES); init = 0.0)

    # Assemble every point of this run: paths first, then grid.
    kind  = String[]; pname = String[]; depth = Float64[]
    P_GPa = Float64[]; T_K = Float64[]
    for p in paths
        n = length(p.P_GPa)
        append!(kind, fill("path", n)); append!(pname, fill(p.name, n))
        append!(depth, p.depth_km); append!(P_GPa, p.P_GPa); append!(T_K, p.T_K)
    end
    if DO_GRID
        for P in GRID_P_GPA, T in GRID_T_K
            push!(kind, "grid"); push!(pname, "grid"); push!(depth, NaN)
            push!(P_GPa, P); push!(T_K, T)
        end
    end

    @printf("  [run ] %-30s db=%-10s MgO=%5.2f K2O=%4.2f  %d points%s\n",
            run_id, key, row.MgO, row.K2O, length(P_GPa),
            isempty(dropped) ? "" : "  (dropped: $(join(dropped, ",")) = $(round(dropped_wt, digits=2)) wt%)")
    flush(stdout)

    data = magemin_db(key)
    outs = multi_point_minimization(10.0 .* P_GPa, T_K .- 273.15, data;
                                    X = X, Xoxides = Xox, sys_in = "wt",
                                    rm_list = RM_LISTS[key], progressbar = false)

    pts = NamedTuple[]
    phs = NamedTuple[]
    for (i, out) in enumerate(outs)
        s = split_point(out)
        assemblage = join(sort([p.phase for p in s.phases if p.frac_sys > 1e-6]), "+")
        push!(pts, (run_id = run_id, db = key, kind = kind[i], path = pname[i],
                    depth_km = depth[i], P_GPa = P_GPa[i], T_K = T_K[i],
                    status = out.status,
                    h2o_solid = s.h2o_solid,
                    solid_wt = s.solid_wt, fluid_wt = s.fluid_wt,
                    melt_wt = s.melt_wt, h2o_melt_sys = s.h2o_melt_sys,
                    saturated = s.fluid_wt > FLUID_PRESENT_THRESHOLD,
                    melt_present = s.melt_wt > MELT_PRESENT_THRESHOLD,
                    n_phases = length(s.phases),
                    assemblage = assemblage,
                    dataset = hasproperty(out, :dataset) ? out.dataset : ""))
        for p in s.phases
            p.kind == "solid" || p.kind == "melt" || continue
            # phase_wt_solid: wt% of the solid rock (NaN for melt)
            # h2o_contrib:    wt% H2O this phase holds per 100 g solid rock
            ws = p.kind == "solid" && s.m_solid > 0 ? 100 * p.frac_sys / s.m_solid : NaN
            push!(phs, (run_id = run_id, kind = kind[i], path = pname[i],
                        depth_km = depth[i], P_GPa = P_GPa[i], T_K = T_K[i],
                        phase = p.phase, model = p.model, phase_kind = p.kind,
                        phase_wt_solid = ws,
                        h2o_in_phase = 100 * p.h2o_in_phase,
                        h2o_contrib = p.kind == "solid" ? ws * p.h2o_in_phase : NaN))
        end
    end

    df_pts = DataFrame(pts); df_ph = DataFrame(phs)
    # Atomic: write to .tmp then rename, so a killed task never leaves a
    # half-written file that the restart logic would treat as done.
    CSV.write(f_ph * ".tmp", df_ph);  mv(f_ph * ".tmp", f_ph, force = true)
    CSV.write(f_pts * ".tmp", df_pts); mv(f_pts * ".tmp", f_pts, force = true)

    n = nrow(df_pts)
    n_fail  = count(>=(3), df_pts.status)
    n_unsat = count(!, df_pts.saturated)
    n_melt  = count(df_pts.melt_present)
    @printf("  [done] %-26s failed %d/%d  undersaturated %d  melt %d\n",
            run_id, n_fail, n, n_unsat, n_melt)
    if n_unsat > 0.1 * n
        @warn "$run_id: >10% of points undersaturated at H2O_EXCESS=$(H2O_EXCESS) wt%"
    end
    flush(stdout)
    return :ok
end

# =============================================================================
# MAIN
# =============================================================================

println("="^64)
println("MAGEMin manifest driver ($VERSION_TAG)")
println("="^64)
println("  MAGEMin_C    : ", pkgversion(MAGEMin_C))
println("  manifest     : ", OPTS["manifest"])
println("  paths        : ", OPTS["paths"])
println("  output       : ", OUTPUT_DIR)
println("  H2O excess   : $H2O_EXCESS wt%")
println("  grid         : ", DO_GRID ? "$(length(GRID_P_GPA)) P x $(length(GRID_T_K)) T" : "off")
println("  threads      : ", Threads.nthreads())
flush(stdout)

paths = load_paths(OPTS["paths"])
println("  P-T paths    : ", isempty(paths) ? "none" :
        join(["$(p.name) ($(length(p.P_GPa)) pts)" for p in paths], ", "))

manifest = CSV.read(OPTS["manifest"], DataFrame)
sel = trues(nrow(manifest))
istrue(OPTS["pilot"]) && (sel .&= manifest.pilot .== true)
isempty(OPTS["db"]) || (sel .&= manifest.db .== OPTS["db"])
isempty(OPTS["scenario"]) ||
    (sel .&= [OPTS["scenario"] in split(string(x), ";") for x in manifest.scenarios])
if !isempty(OPTS["rows"])
    a, b = parse.(Int, split(OPTS["rows"], ":"))
    r = falses(nrow(manifest)); r[a:min(b, nrow(manifest))] .= true
    sel .&= r
end
idx = findall(sel)

task_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "-1"))
n_tasks = parse(Int, get(ENV, "SLURM_ARRAY_TASK_COUNT", "1"))
if task_id >= 0
    idx = [i for (k, i) in enumerate(idx) if (k - 1) % n_tasks == task_id]
    println("  SLURM array task $task_id / $n_tasks")
end
println("  rows to run  : $(length(idx)) of $(nrow(manifest))")
flush(stdout)

# Rows are run one at a time; MAGEMin threads over the points within a row.
results = Symbol[]
for i in idx
    r = try
        run_one(manifest[i, :], paths)
    catch e
        @warn "row $i ($(manifest.run_id[i])) failed" exception = (e, catch_backtrace())
        :failed
    end
    push!(results, r)
end

for d in values(DBS)
    Finalize_MAGEMin(d)
end

println("\n", "="^64)
println("  ok $(count(==(:ok), results))   skipped $(count(==(:skipped), results))",
        "   failed $(count(==(:failed), results))")
println("  per-run tables in: $OUTPUT_DIR")
println("  NEXT: python h2o_lookup/postprocess_v4.py --manifest $(OPTS["manifest"]) --runs $OUTPUT_DIR")
println("="^64)

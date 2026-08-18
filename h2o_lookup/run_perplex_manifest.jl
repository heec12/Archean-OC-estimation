# run_perplex_manifest.jl
# =======================
# Stage 2 of the Archean oceanic crust bound-H2O pipeline.
# Replaces 5H2O_test_v2_parallel.jl.
#
# WHAT CHANGED AND WHY
# --------------------
# The old script looped over two SCENARIOS x N ensemble samples and then
# collapsed the sample dimension into mean / std / p05 / p50 / p95 lookup
# tables. Under the new 2-D design that averaging is exactly wrong: it would
# smear MgO 10 and MgO 30 into one number and delete the response curve before
# it could be seen.
#
# So this script does NO statistics. It writes one long-format table per run
# and leaves every reduction to postprocess_response.py. Storage is trivial
# (40 x 40 x ~200 runs) and it means the reduction can be changed without
# re-running any minimisation.
#
# Other changes:
#   * Reads composition_manifest.csv (metadata + 7 oxides per row) instead of
#     ensemble_<scenario>.csv. Scratch dirs are keyed on run_id, not on a bare
#     loop index, so a rerun of one design cell cannot collide with another.
#   * SCENARIOS is gone. f is applied in post-processing, so upper and lower
#     crust are just rows in the manifest.
#   * H2O_EXCESS = 20.0, not 5.0. At MgO 28-30 the carriers are serpentine,
#     brucite, talc and chlorite, and a fully hydrated ultramafic rock holds
#     12-15 wt% H2O. With 5 wt% added the rock is water-UNDERSATURATED, every
#     drop gets bound, and bound-H2O pins at 5.0 over a large part of P-T. The
#     curve would rise with MgO and then flatten, and the flattening would be
#     the input cap masquerading as petrology. 20 wt% clears the ceiling.
#   * Saturation is CHECKED, not assumed. Every P-T cell records the free
#     fluid (h2oL) abundance. A cell with no free fluid is undersaturated and
#     its bound-H2O is censored, not a capacity -- post-processing masks those.
#   * Solution model list extended for the ultramafic end (serpentine, brucite,
#     talc). Without them Perple_X does not error, it just returns less water,
#     and that would read as a real compositional signal.
#   * Restartable: existing per-run output is skipped. Supports SLURM array
#     jobs (rows split round-robin across tasks) as well as plain threading.
#
# USAGE
# -----
#   julia --threads 8 run_perplex_manifest.jl --manifest path/to/manifest.csv
#
#   # SLURM array (preferred: ~200 runs will have failures, and array tasks
#   # let you rerun only the holes)
#   sbatch submit_perplex_array.sbatch
#
# Versioning: PERPLEX_VERSION is parsed from this file's own "_vN" token if
# present, else from --version, else "v3". It tags scratch dirs and the output
# directory only -- the manifest is a design artifact and is unversioned.

using StatGeochem
using StatGeochem.Perple_X_jll        # explicit import: `using StatGeochem` alone is not enough
using CSV, DataFrames, Printf

# =============================================================================
# SETTINGS
# =============================================================================

const OXIDE_COLS = ["SiO2_pct", "TiO2_pct", "Al2O3_pct",
                    "FeOtot_pct", "MgO_pct", "CaO_pct", "Na2O_pct"]

# Perple_X component names in hp62ver.dat are CASE-SENSITIVE MIXED CASE.
# Uppercase (SIO2) breaks the build silently.
const PERPLEX_COMPONENTS = ["SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "H2O"]

# Excess H2O, wt% added to every anhydrous composition. Must exceed the maximum
# bound-H2O capacity of the most hydrous assemblage in the design (fully
# serpentinised + brucite-bearing ultramafic, ~15 wt%) so that free fluid is
# present at every P-T cell and the query returns a CAPACITY rather than
# "how much of my input got consumed".
const H2O_EXCESS = 20.0

# Below this free-fluid wt% a cell is treated as undersaturated / censored.
const FLUID_PRESENT_THRESHOLD = 0.01

# P-T grid. P in bar, T in Kelvin.
const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length = 40))
const T_VEC = collect(range(273.0, 1600.0, length = 40))

# Solution models. VERIFIED against perplex_scratch/out1/solution_model.dat
# and hp62ver.dat in the Archean-OC-estimation repo.
#
#   cAmph(G)  -- Green et al. 2016, ds62-calibrated. Replaces Amph(DPW), which
#                triggers an interactive ver017 Y/N prompt and crashes on EOF
#                when stdin is piped.
#   Sp(WPC)   -- replaces Sp(HGP), which needs ds6.33 make-definitions absent
#                from hp62ver.dat.
#   Atg(PN)   -- antigorite, Padron-Navarta et al. 2013. REQUIRED for the
#                high-MgO cells. Endmembers atg / fatg / atgts: atg is in
#                hp62ver.dat directly, fatg and atgts resolve via the
#                make-definitions at hp62ver.dat lines 155 and 158, so this
#                model will NOT be silently rejected the way Sp(HGP) was.
#   T         -- talc (abbreviation Tlc; endmembers ta / fta / tats).
#   B         -- brucite (endmembers br / fbr). This IS a solution model, not
#                a bare endmember. Matters at the ultramafic end: fully
#                hydrated dunite is serpentine + brucite, and leaving B out
#                costs real bound H2O in exactly the high-MgO cells the design
#                exists to probe.
#   Chl(W)    -- chlorite.
#
# Ep(HP) and Pheng(HP) are silently REJECTED in NCFMAST (they need O2 and K2O).
# They are omitted rather than listed so the phase list is honest. NOTE: no
# K2O means no phengite, and phengite is the dominant deep H2O carrier in
# metabasites -- worth a separate KNCFMASH test run before concluding the deep
# tail is negligible.
#
# oAmph(DP) (orthoamphibole) also exists in solution_model.dat and is relevant
# at ultramafic compositions, but EXCLUDES below removes ged and fanth, which
# are its endmembers -- adding oAmph(DP) without revisiting EXCLUDES would
# give a crippled model. Left out deliberately; revisit if anthophyllite turns
# up as a carrier in the sensitivity ladder.
const SOLUTION_PHASES = join([
    "O(HGP)",       # olivine
    "Cpx(HGP)",
    "Opx(HGP)",
    "Gt(HGP)",
    "Fsp(HGP)",
    "Sp(WPC)",
    "Chl(W)",
    "cAmph(G)",
    "Atg(PN)",      # antigorite -- ultramafic end
    "T",            # talc
    "B",            # brucite
], "\n") * "\n"

const EXCLUDES = "ts\nparg\ngl\nged\nfanth\n"

# =============================================================================
# PATHS
# =============================================================================

const BASE = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation"

function parse_args()
    opts = Dict{String,String}(
        "manifest" => joinpath(BASE, "design_outputs", "composition_manifest.csv"),
        "outdir"   => "",
        "scratch"  => joinpath(BASE, "perplex_ensemble"),
        "version"  => "",
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

const OPTS = parse_args()

const PERPLEX_VERSION = let
    if !isempty(OPTS["version"])
        OPTS["version"]
    else
        m = match(r"_v(\d+)", basename(@__FILE__))
        m === nothing ? "v3" : "v" * m.captures[1]
    end
end

const MANIFEST_PATH = OPTS["manifest"]
const SCRATCH_DIR   = OPTS["scratch"]

# Scratch dirs are deleted after each successful run by default. Set
# --keep-scratch true only when debugging one composition: keeping all of them
# needs several TB.
const KEEP_SCRATCH = lowercase(get(OPTS, "keep-scratch", "false")) in ("true", "1", "yes")
const OUTPUT_DIR    = isempty(OPTS["outdir"]) ?
    joinpath(BASE, "h2o_runs", PERPLEX_VERSION) : OPTS["outdir"]

mkpath(OUTPUT_DIR)
mkpath(SCRATCH_DIR)

# =============================================================================
# POINT PARSER
# =============================================================================
"""
    parse_point(point_str)

Returns (h2o_solid_wt, fluid_wt, n_phases).

h2o_solid_wt is read from the "Solid Only" H2O wt% column of Perple_X's Bulk
Composition block. This is the fix from the earlier pipeline: the free fluid
phase is named `h2oL`, not "fluid" or "H2O", so subtracting a phase called
"fluid" silently subtracted nothing and bound-H2O came back inflated by the
whole free-fluid budget.

When no free fluid is stable Perple_X prints only one set of columns (no
"Solid Only" block), which is itself the undersaturation signal.

fluid_wt is the h2oL abundance from the phase block, used as an independent
saturation check.
"""
function parse_point(point_str::String)
    isempty(strip(point_str)) && return (NaN, NaN, 0)

    h2o_solid = NaN
    fluid_wt  = 0.0
    n_phases  = 0

    in_phase_block = false
    in_bulk_block  = false
    has_solid_only = false

    for line in split(point_str, "\n")

        # ---- phase block ----
        if occursin("Phase Compositions (weight percentages)", line)
            in_phase_block = true
            continue
        end
        if in_phase_block && (occursin("Phase speciation", line) ||
                              occursin("Bulk Composition", line))
            in_phase_block = false
        end
        if in_phase_block
            tokens = split(strip(line))
            # name + wt% + vol% + mol% + mol + 8 components = 13 tokens
            if length(tokens) == 13
                wt = tryparse(Float64, tokens[2])
                if wt !== nothing
                    n_phases += 1
                    if tokens[1] == "h2oL"
                        fluid_wt = wt
                    end
                end
            end
        end

        # ---- bulk composition block ----
        if occursin("Bulk Composition", line)
            in_bulk_block = true
            continue
        end
        if in_bulk_block
            if occursin("Solid Only", line)
                has_solid_only = true
            end
            tokens = split(strip(line))
            if length(tokens) >= 4 && tokens[1] == "H2O"
                if has_solid_only && length(tokens) >= 8
                    # name + 4 complete-assemblage cols + 4 solid-only cols
                    #   -> solid-only wt% is token 8
                    v = tryparse(Float64, tokens[8])
                    h2o_solid = v === nothing ? NaN : v
                else
                    v = tryparse(Float64, tokens[4])
                    h2o_solid = v === nothing ? NaN : v
                end
            end
        end
    end

    return (h2o_solid, fluid_wt, n_phases)
end

# =============================================================================
# ONE RUN
# =============================================================================
"""
Long format, one row per P-T cell. Long format on purpose: the matrix layout
with CR-only line endings is a TerraFERMA input format, produced at the very
end of post-processing, not an intermediate.
"""
function run_one(row, scratch_root::String, out_dir::String)
    run_id = String(row.run_id)
    out_path = joinpath(out_dir, "h2o_$(run_id).csv")

    if isfile(out_path)
        println("  [skip] $run_id already done")
        flush(stdout)
        return :skipped
    end

    comp = Float64[row[c] for c in OXIDE_COLS]
    push!(comp, H2O_EXCESS)

    # Per-run AND per-version scratch: Perple_X writes fixed-name working
    # files, so parallel runs sharing a directory clobber each other.
    scratchdir = joinpath(scratch_root, "$(PERPLEX_VERSION)_$(run_id)")
    mkpath(scratchdir)

    @printf("  [run ] %s  MgO=%.2f  Al/Ti=%.1f  layer=%s\n",
            run_id, row.MgO_pct, row.al_ti_upper, row.layer)
    flush(stdout)

    try
        perplex_configure_pseudosection(
            scratchdir,
            comp,
            PERPLEX_COMPONENTS,
            (minimum(P_VEC), maximum(P_VEC)),
            (minimum(T_VEC), maximum(T_VEC)),
            dataset          = "hp62ver.dat",
            solution_phases  = SOLUTION_PHASES,
            excludes         = EXCLUDES,
            fluid_eos        = 5,
        )
    catch e
        @warn "  configure failed for $run_id" exception = e
        return :failed
    end

    rows = NamedTuple[]
    n_bad = 0
    for P in P_VEC, T in T_VEC
        h2o_solid, fluid_wt, n_phases = try
            parse_point(perplex_query_point(scratchdir, P, T))
        catch e
            n_bad += 1
            (NaN, NaN, 0)
        end
        push!(rows, (
            run_id      = run_id,
            P_GPa       = P / 10000.0,
            T_K         = T,
            h2o_solid   = h2o_solid,
            fluid_wt    = fluid_wt,
            saturated   = (!isnan(fluid_wt) && fluid_wt > FLUID_PRESENT_THRESHOLD),
            n_phases    = n_phases,
        ))
    end

    df = DataFrame(rows)
    CSV.write(out_path, df)

    # Delete the Perple_X working directory once the table is safely written.
    #
    # A scratch dir is ~11 MB, so 192 runs is only a couple of GB -- not a
    # crisis, but the v1 and v2 ensembles left 800 stale directories behind and
    # there is no reason to accumulate more. The extracted P-T table is the
    # product; the working files are regenerable and nothing downstream reads
    # them.
    #
    # Order matters: CSV.write comes first, so a crash mid-cleanup costs a
    # scratch dir, never a result. Pass --keep-scratch true to retain them when
    # debugging a specific composition.
    if KEEP_SCRATCH
        @info "  keeping scratch for $run_id at $scratchdir"
    else
        try
            rm(scratchdir, recursive = true, force = true)
        catch e
            @warn "  could not remove scratch for $run_id" exception = e
        end
    end

    n_nan = count(isnan, df.h2o_solid)
    n_unsat = count(!, df.saturated)
    @printf("  [done] %s  NaN %d/%d  undersaturated %d/%d\n",
            run_id, n_nan, nrow(df), n_unsat, nrow(df))
    if n_unsat > 0.1 * nrow(df)
        @warn "  $run_id: >10% of cells undersaturated at H2O_EXCESS=$(H2O_EXCESS) wt%. " *
              "Raise H2O_EXCESS -- bound H2O in those cells is censored, not a capacity."
    end
    flush(stdout)

    return :ok
end

# =============================================================================
# MAIN
# =============================================================================

println("="^62)
println("Perple_X manifest driver")
println("="^62)
println("  version      : $PERPLEX_VERSION")
println("  manifest     : $MANIFEST_PATH")
println("  output       : $OUTPUT_DIR")
println("  scratch      : $SCRATCH_DIR")
println("  H2O excess   : $H2O_EXCESS wt%")
println("  grid         : $(length(P_VEC)) P x $(length(T_VEC)) T")
println("  threads      : $(Threads.nthreads())")
println("  solutions    : ", replace(strip(SOLUTION_PHASES), "\n" => ", "))
flush(stdout)

println("  keep scratch : $KEEP_SCRATCH")

# Preflight. Measured size of one 8-component, 40x40 Perple_X scratch dir is
# about 11 MB (v1 ensemble on Phoenix), so budget ~15 MB per run. With cleanup
# on, only a few exist at once; with --keep-scratch, all of them do.
const SCRATCH_GB_PER_RUN = 0.015

try
    avail_kb = parse(Int, split(readchomp(`df -P -k $SCRATCH_DIR`), '\n')[2] |> x -> split(x)[4])
    avail_gb = avail_kb / 1024 / 1024
    @printf("  free scratch : %.1f GB\n", avail_gb)
    needed = KEEP_SCRATCH ? SCRATCH_GB_PER_RUN * 250 :
                            SCRATCH_GB_PER_RUN * Threads.nthreads() * 4
    if avail_gb < max(needed, 5.0)
        @warn @sprintf("Only %.1f GB free (need ~%.1f GB plus room for output). " *
                       "Check `pace-quota` before running.", avail_gb, needed)
    end
catch
    println("  free scratch : (could not determine)")
end

manifest = CSV.read(MANIFEST_PATH, DataFrame)
println("  manifest rows: $(nrow(manifest))")

# SLURM array support: split rows round-robin so each task gets a mix of
# cheap and expensive compositions rather than a contiguous block.
task_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "-1"))
n_tasks = parse(Int, get(ENV, "SLURM_ARRAY_TASK_COUNT", "1"))

row_indices = if task_id >= 0
    idx = [i for i in 1:nrow(manifest) if (i - 1) % n_tasks == task_id]
    println("  SLURM array task $task_id / $n_tasks -> $(length(idx)) rows")
    idx
else
    collect(1:nrow(manifest))
end
flush(stdout)

results = Vector{Symbol}(undef, length(row_indices))

Threads.@threads for k in eachindex(row_indices)
    i = row_indices[k]
    results[k] = try
        run_one(manifest[i, :], SCRATCH_DIR, OUTPUT_DIR)
    catch e
        @warn "  row $i raised" exception = e
        :failed
    end
end

n_ok      = count(==(:ok), results)
n_skipped = count(==(:skipped), results)
n_failed  = count(==(:failed), results)

println("\n" * "="^62)
println("  ok $n_ok   skipped $n_skipped   failed $n_failed")
println("  per-run tables in: $OUTPUT_DIR")
println("\nNEXT: python postprocess_response.py --manifest $MANIFEST_PATH \\")
println("          --runs $OUTPUT_DIR --output ./response_outputs")
println("="^62)
flush(stdout)

if n_failed > 0
    println("\nRerun only the failures: completed runs are skipped on restart,")
    println("so simply resubmitting the same array job fills the holes.")
end

"""
diagnose_parser.jl
==================
Tests the ACTUAL lookup quantity (parse_bound_h2o on perplex_query_point output),
not mineral modes. If sample #44's *parsed* bound H2O shows a local max at the
spike column — even though its modes are smooth — the bug is in the parser, in
the bulk-minus-free-fluid subtraction.

It (1) prints parsed bound H2O for the spike column vs neighbours over high P,
and (2) dumps the raw werami point output at a spike cell and a neighbour cell
so you can see how the free fluid is reported and why detection fails.
"""

using StatGeochem

const SCRATCH = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/perplex_ensemble/v1_homogeneous_crust_sample_44"
const P_VEC   = collect(range(0.0001 * 10000, 8.0 * 10000, length=40))  # bar
const T_VEC   = collect(range(273.0, 1600.0, length=40))                # K
const T_SPIKE_C = 612.3
const P_DUMP_GPA = 6.0

# ---- parse_bound_h2o : copied VERBATIM from 5H2O_test_v2_parallel.jl ----------
function parse_bound_h2o(point_str::String)
    isempty(strip(point_str)) && return NaN
    phase_h2o = Dict{String, Float64}()
    bulk_h2o = NaN
    in_phase_block = false
    for line in split(point_str, "\n")
        if occursin("Phase Compositions (weight percentages)", line)
            in_phase_block = true; continue
        end
        if in_phase_block && occursin("Phase speciation", line)
            in_phase_block = false
        end
        if in_phase_block
            tokens = split(strip(line))
            if length(tokens) == 13
                wt_pct = tryparse(Float64, tokens[2])
                h2o_in_phase = tryparse(Float64, tokens[end])
                if wt_pct !== nothing && h2o_in_phase !== nothing
                    phase_h2o[tokens[1]] = (wt_pct / 100.0) * h2o_in_phase
                end
            end
        end
        m = match(r"^\s+H2O\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        m !== nothing && (bulk_h2o = parse(Float64, m[1]))
    end
    free_fluid_contrib = get(phase_h2o, "fluid", get(phase_h2o, "H2O", 0.0))
    bound_h2o = isnan(bulk_h2o) ? NaN : bulk_h2o - free_fluid_contrib
    return max(0.0, bound_h2o)
end
# ------------------------------------------------------------------------------

j = argmin(abs.((T_VEC .- 273.15) .- T_SPIKE_C))
cols = [j-1, j, j+1]
println("Parsed bound H2O (wt%) — the actual lookup quantity, sample #44")
println("Columns: ", round.(T_VEC[cols] .- 273.15, digits=1), " °C   (middle = spike)\n")
println("  P(GPa) | ", join([rpad("$(round(T_VEC[c]-273.15,digits=0))C", 10) for c in cols], ""))
for i in 1:length(P_VEC)
    (P_VEC[i] / 10000) < 4.0 && continue
    vals = [parse_bound_h2o(perplex_query_point(SCRATCH, P_VEC[i], T_VEC[c])) for c in cols]
    flag = (vals[2] - max(vals[1], vals[3])) > 0.5 ? "  <-- spike local max" : ""
    println("  ", rpad(round(P_VEC[i]/10000, digits=2), 8), "| ",
            join(rpad.(round.(vals, digits=3), 10), ""), flag)
end

# ---- raw dump: see how the fluid is reported at a spike cell vs a neighbour ---
i_dump = argmin(abs.((P_VEC ./ 10000) .- P_DUMP_GPA))
for (label, c) in (("SPIKE  $(T_SPIKE_C)C", j), ("NEIGHBOUR (hotter)", j+1))
    println("\n" * "="^70)
    println("RAW werami output — $label, P = $(round(P_VEC[i_dump]/10000,digits=2)) GPa")
    println("parsed bound H2O = ",
            round(parse_bound_h2o(perplex_query_point(SCRATCH, P_VEC[i_dump], T_VEC[c])), digits=3), " wt%")
    println("="^70)
    println(perplex_query_point(SCRATCH, P_VEC[i_dump], T_VEC[c]))
end

import glob, sys
import pandas as pd

rows, unreadable = [], []
for f in sorted(glob.glob("h2o_runs/v3/h2o_*.csv")):
    try:
        d = pd.read_csv(f)
    except Exception as e:
        unreadable.append((f, str(e)))
        continue
    sat = d["saturated"].astype(str).str.lower().isin(["true", "1"])
    rows.append(dict(run=d.run_id.iloc[0], n=len(d), sat=sat.mean(),
                     hmin=d.h2o_solid.min(), hmax=d.h2o_solid.max(),
                     nan=d.h2o_solid.isna().sum()))

s = pd.DataFrame(rows)
s["layer"] = s.run.str.split("_").str[-1]
bad = s[(s.n != 1600) | (s.sat < 0.5) | (s.hmin < 0) | (s.hmax > 20) | (s.nan > 0)]

print(f"{len(s)} readable runs, {len(unreadable)} unreadable, {len(bad)} flagged")
for f, e in unreadable:
    print("  UNREADABLE:", f, e)
if len(bad):
    print(bad.to_string(index=False))
print(s.groupby("layer")[["sat", "hmin", "hmax"]].describe().round(2).T)

sys.exit(1 if (len(bad) or unreadable or len(s) != 192) else 0)

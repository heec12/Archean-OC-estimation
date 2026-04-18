"""
reformat_h2o_lookup.py
======================
Converts Julia ensemble output CSVs to the target lookup table format:
  - Empty top-left cell
  - Header row: T in Kelvin
  - First column: P in GPa
  - Values: bound H2O wt% (negatives clamped to 0)
  - CR-only line endings
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_DIR  = Path(".")
OUTPUT_DIR = Path(".")

SCENARIOS  = ["homogeneous_crust", "layered_cumulate_lower_crust"]
STAT_TYPES = ["mean", "std"]

def write_cr_only(df: pd.DataFrame, out_path: Path):
    """Write DataFrame with P_bar column to CR-only CSV with P in GPa."""
    lines = []

    # T column names (already in Kelvin from Julia script)
    t_cols = [c for c in df.columns if c != 'P_bar']

    # Header: empty cell + T values (strip trailing zeros)
    t_labels = []
    for c in t_cols:
        t = float(str(c).replace('C','')) + 273.15 if str(c).endswith('C') else float(c)
        s = f"{t:.3f}".rstrip('0').rstrip('.')
        t_labels.append(s)
    lines.append(',' + ','.join(t_labels))

    # Data rows: P converted from bar to GPa
    for _, row in df.iterrows():
        p_gpa = row['P_bar'] / 10000.0
        p_str = f"{p_gpa:.4f}".rstrip('0').rstrip('.')
        vals  = [max(0.0, v) for v in row[t_cols]]   # clamp negatives
        lines.append(p_str + ',' + ','.join(f"{v:.6g}" for v in vals))

    out_path.write_bytes('\r'.join(lines).encode('ascii'))
    print(f"  Written: {out_path}  ({len(df)} rows × {len(t_cols)} cols)")

def main():
    for scenario in SCENARIOS:
        for stat in STAT_TYPES:
            in_path  = INPUT_DIR  / f"h2o_bound_{stat}_{scenario}.csv"
            out_path = OUTPUT_DIR / f"h2o_bound_{stat}_{scenario}_formatted.csv"

            if not in_path.exists():
                print(f"  SKIPPING (not found): {in_path}")
                continue

            print(f"\nProcessing: {in_path.name}")
            df = pd.read_csv(in_path)
            write_cr_only(df, out_path)

if __name__ == "__main__":
    main()


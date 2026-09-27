"""Tighten column-hinge brackets that jumped from 4.4% to 6.0% or 6.0% to 8.0%."""

from __future__ import annotations

import json

import opensees_moment_frame as mf
import refine_moment_bracket as rb
from refine_moment_bracket import bracket_one

mf.COLUMN_HINGES = True
rb.LEVELS = (0.046, 0.050, 0.054, 0.058, 0.064, 0.070, 0.076)

SRC = mf.OUT / "column_hinge_rows.json"
motions = dict(mf.load_records())
rows = json.loads(SRC.read_text(encoding="utf-8"))
for row in rows:
    level = row.get("committing_level")
    if row.get("status") != "committed" or level is None or level < 0.06:
        row["drift_fine"] = row.get("drift_at_cut")
        row["level_fine"] = level
        row["recover_fine"] = row.get("last_recover_level")
        continue
    hit = bracket_one(row["nstory"], motions[row["id"]], row["scale"])
    row["drift_fine"] = hit.get("drift_at_cut", row.get("drift_at_cut"))
    row["level_fine"] = hit.get("committing_level", level)
    row["recover_fine"] = hit.get("last_recover_level", row.get("last_recover_level"))
    row["fine_status"] = hit.get("status")
    print(
        f"{row['nstory']}s {row['id']} coarse={level} fine_level={row['level_fine']} "
        f"drift={row['drift_fine']} recover={row['recover_fine']}",
        flush=True,
    )
    SRC.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("DONE", flush=True)

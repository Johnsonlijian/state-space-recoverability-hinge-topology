# Reproducible runbook

## Environment

Use Python 3.12 with NumPy, Matplotlib, and OpenSeesPy 3.8 or a compatible
release. The OpenSees model is deterministic for a fixed component file and
scale grid. The current manuscript used the local Windows OpenSeesPy runtime
and stored the completed derived tables in `results/`.

## Inputs

Obtain the FEMA P695 far-field component files through a legally permitted
source and place the 21 even-numbered `FF*.txt` files in
`data/fema_p695/`. The package intentionally contains only the source-data
manifest and the derived results.

## Commands

From the repository root:

```powershell
python src/explore_dynamic_full.py
python src/static_continuation.py
python src/paired_comparison.py
python src/summarize_multibay.py
python src/make_multibay_figure.py
python src/run_study.py
python src/sdof_recorded.py
python src/opensees_moment_frame.py
python src/make_shear_frame_figure.py
python src/make_frame_figure.py
python src/make_record_interaction_figure.py
python src/make_graphical_abstract.py
```

The first two commands regenerate the dynamic and static derived tables.
The figure commands rebuild the editable figure masters in `figures/`,
including `fig5_frame_commitment.*` from `results/frame_rows.json`.
Existing JSON files can be used for a no-download verification of the
manuscript numbers.

The baseline and multi-bay models are controlled idealizations. They are not a
code-designed building and the stored results do not establish experimental
validation or a population-level fragility model. The public archive DOI must
be added to the manuscript data citation after a permitted repository deposit.

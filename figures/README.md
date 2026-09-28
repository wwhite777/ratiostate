# Recreate and edit the four reported figures

The accepted conceptual Figure 1 is supplied as `system_overview.png`, vector `system_overview.pdf` and `system_overview.svg`, plus the native editable `system_overview.pptx`. `system_overview_caption_alt.md` provides its caption and accessible description. The original private conceptual-figure builder was not included because it depends on manuscript source files outside this code/data export; the editable source and rendered files are supplied instead.

For reported Figures 2–4, use the unchanged scripts and adjacent frozen inputs here. Tested in the existing Linux environment with Python 3.10.12, NumPy 1.26.4 and Matplotlib 3.10.7; a clean installation was not tested. From this directory, direct outputs to fresh writable paths outside the repository:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -I -B make_figures_v3.py --data-dir plot_data_v1 --output-dir /fresh/result-figures
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -I -B make_figure3_v5.py --output-dir /fresh/accepted-figure4
```

The first command's `primary_contrasts_v3` is manuscript Figure 2 and `score_dispersion_v3` is Figure 3. Its legacy `position_mass_score_v3` is superseded and should not be used. The second command produces accepted manuscript Figure 4 (`position_mass_score_v5`). Replace `/fresh/...` with unused absolute output directories. The scripts and all required local plotting inputs are byte-identical to the v5 figure reproduction resource. Figure 4's selected normalizer mass ratios are not query-weighted denominator errors or causal mediation estimates.

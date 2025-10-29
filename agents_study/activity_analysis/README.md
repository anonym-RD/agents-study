## Installation

First clone and cd into the repo.

We recommend you use first setup a miniconda or virtual env.
```bash
## Optional using conda
conda create -n agents python=3.12
conda activate agents
####
## CD INTO DIR
pip install -e .
```

## Generating Paper Figures

Run these scripts to generate figures and tables used in the paper:

### Main Analysis Scripts

- `spoiler/analysis/pape_vars.py` -> `plots/gen_vars.tex` (Some LaTeX variables for prose)
- `spoiler/analysis/repo_stars_table.py` -> `plots/repo_analysis/repo_stars_table.tex` (Table 2 on num of stars per repo)
- `spoiler/analysis/main_metrics_overview.py` -> Generates several figures. Some of them that are in paper:
  - `plots/main_metrics/change_size.pdf` (Figure: change size & addition ratio)
  - `plots/main_metrics/engagement.pdf` (Figure: comments & reviews)
- `spoiler/analysis/language_analysis_files.py` -> `plots/language_analysis/language_presence_by_files.pdf` (Figure: file types)
- `spoiler/analysis/merge_rate_stratified.py` -> `plots/main_metrics/merge_rate_stratified.pdf` (Figure: merge rate by stars)
- `spoiler/analysis/linked_issues_analysis.py` -> `plots/linked_issues/linked_issues_fraction.pdf` (Figure: linked issues)
- `spoiler/analysis/time_to_merge_stratified.py` -> `plots/time_to_merge_stratified/time_to_merge_stratified.pdf` (Figure: time to merge by stars)

### Helper Scripts

- `spoiler/analysis/load_hf_data_polars.py` - Loads data as lazy polars frames, caches to disk after download from HF
- `spoiler/util` - Common plotting utilities and helpers for generating tex files.

## Running Analysis Scripts

All analysis scripts can be run from the project root. For example:
```bash
conda run -n spoiler python -m spoiler.analysis.main_metrics_overview
```
Which makes some of the figures descrived above.
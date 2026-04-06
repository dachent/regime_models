# regime_models

`regime_models` is a monorepo for regime-model research, including faithful reproductions of published models, derived variants, and original models created in-house.

The goal is to keep each model folder self-contained, reproducible, and reviewable:

- runnable code
- one canonical `README.md`
- explicit data provenance
- validation against the original paper or source model
- committed summary artifacts such as tables and charts

## Folder Naming

Each model folder name must follow:

`<prefix>_<lead-author><year>_<slug>`

Allowed prefixes:

- `reprod_` for a direct reproduction of a published or third-party model
- `derived_` for a materially modified variant of an existing model
- `created_` for a new model authored in this repo

Rules:

- use lowercase ASCII only
- use underscores, not spaces or hyphens
- use the lead author or originator surname plus source year
- end with a short descriptive slug
- folder names must be unique within the repo

Example:

- `reprod_mulliner2025_regimes`

## Repository Layout

Each top-level model folder owns its own implementation, memo, and artifacts.

Recommended structure:

```text
regime_models/
  README.md
  <model-folder>/
    README.md
    requirements.txt
    data_access.py
    model_core.py
    regime_model_final.py
    scenario_tester.py
    artifacts/
      *.csv
      figures/
```

Raw downloaded vendor data is not committed. Workspace-local caches are allowed during execution but must be ignored by Git.

## Data Access Policy

All models in this repo must obtain data through:

- the current `dachent/mdt` source set
- or direct provider endpoints that are consistent with it

That means:

- no manual pre-download steps for required inputs
- no dependence on ad hoc local Excel or CSV files
- no legacy `yf_marketdata` assumption for Yahoo flows
- Yahoo-family data should use the current direct yfinance-compatible flow
- FRED, Macrotrends, EIA, Treasury, and Fama-French inputs should use direct provider endpoints or helpers aligned with `mdt`

If a provider is unreliable from a given environment, code may use a transport fallback, but the underlying source and semantics must stay consistent with the approved provider.

## Minimum Requirements For Every Model Folder

Every model folder must include:

- `README.md` as the single source of truth for the model memo
- a runnable entrypoint for the selected production configuration
- source citations with URLs or precise identifiers
- an exact input-data mapping from model variables to providers and endpoints
- reconstruction or derivation notes
- validation results versus the original paper, model, or benchmark
- committed summary artifacts such as charts, metric tables, and scenario tables
- a clear statement of residual gaps, if any

Every model folder must not include:

- committed raw vendor downloads
- duplicate memos in multiple canonical formats
- undocumented manual setup steps for required data

## Review Standard

A model is considered repo-compliant when:

- the code runs from a clean checkout after dependency install
- required inputs are fetched automatically
- the folder README explains the implementation and validation clearly enough for another researcher to audit it
- the committed artifacts match the current code output

## Current Models

- [`reprod_mulliner2025_regimes`](./reprod_mulliner2025_regimes/README.md): reproduction of Mulliner, Harvey, Xia, Fang, and Van Hemert (2025) "Regimes"
- [`reprod_kim2023_dynamic_asset_allocation`](./reprod_kim2023_dynamic_asset_allocation/README.md): reproduction of Kim and Kwon (2023) "Dynamic asset allocation strategy: an economic regime approach"

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
- [`reprod_kwon2022_dynamic_factor_rotation`](./reprod_kwon2022_dynamic_factor_rotation/README.md): reproduction of Kwon (2022) "Dynamic Factor Rotation Strategy: A Business Cycle Approach"
- [`reprod_kim2023_dynamic_asset_allocation`](./reprod_kim2023_dynamic_asset_allocation/README.md): reproduction of Kim and Kwon (2023) "Dynamic asset allocation strategy: an economic regime approach"
- [`reprod_shu2025_asset_specific_regime_forecasts`](./reprod_shu2025_asset_specific_regime_forecasts/README.md): reproduction of Shu, Yu, and Mulvey (2025) "Dynamic asset allocation with asset-specific regime forecasts"

## Cross-Repo Specification and Reproduction Assessment

| Dimension | Mulliner 2025 | Kwon 2022 | Kim & Kwon 2023 | Shu 2025 |
| --- | --- | --- | --- | --- |
| Paper specification completeness | ~90% | ~95% | ~85% | ~70% |
| Key metric match | Q1 Sharpe 0.957 vs 0.95 | IR 0.627 vs 0.626 | IR 0.70 vs 0.74 | MinVar+Regime 0.50 vs 0.94 |
| Reproduction quality | 8/10 | 7.5/10 | 8.5/10 | 4/10 |
| Dominant gap source | Bond proxy (stock-bond correlation sign flip) | Slope at L1 trend-filter kinks | Commodity proxy (PPIACO vs GSCI TR: 3% vol vs 20% vol) | Confounded: data + lambda shortcuts + implementation divergences |
| Gap fixable with public data? | Partially | Possibly (with author guidance) | No (GSCI TR is proprietary) | Unknown (never properly tested) |

### Cross-Cutting Pattern

When the gap between a reproduction and its source paper traces to a single, identified cause, the reproduction succeeds directionally. When multiple sources of error compound and nobody isolates them, the reproduction fails.

| Paper | Primary Gap Type | Confounded Variables | Reproduction Success |
| --- | --- | --- | --- |
| Kwon 2022 | 1 algorithmic ambiguity (slope at kinks) | 1 | High |
| Kim & Kwon 2023 | 1 data proxy (commodity) | 1 | High |
| Mulliner 2025 | 1 data proxy (bond for stock-bond correlation) | 1-2 | Good |
| Shu 2025 | Multiple: data + lambda + implementation | 5+ | Failed |

The three successful reproductions each have a narrow, well-identified residual. The Shu 2025 reproduction has at least five confounded failure modes (data proxy quality, lambda selection shortcuts, return convention, covariance shrinkage, optimizer choice) that were never disentangled. The forward plan for that reproduction calls for stage-by-stage validation to isolate variables --- the approach the other three reproductions effectively followed by having fewer variables to confound in the first place.

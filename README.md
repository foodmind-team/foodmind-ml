# FoodMind ML

FoodMind ML is FoodMind's offline model-development and packaging repository. It validates approved data snapshots, builds hybrid recommendation features, evaluates candidate models, and produces an immutable model package for FoodMind Intelligence.

## Live deployment

The user-facing FoodMind application is deployed at [https://13.229.2.154.sslip.io/](https://13.229.2.154.sslip.io/). This repository runs offline and does not expose a public endpoint; its approved model package is validated and consumed by the private Intelligence runtime.

## Scope

This repository owns data preparation, collaborative-signal research, model training and evaluation, model cards, reproducible experiments, and model-package creation. It does not expose a public API, authenticate users, access the production database, or serve inference requests.

Hard dietary and eligibility constraints are enforced by the Backend before ranking. A model score must never replace those rules.

## Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- An approved local dataset snapshot only when training or building collaborative indexes

Never commit production records, personal identifiers, credentials, or large model/data artifacts.

## Quick start

Install the locked development environment and run the repository checks:

~~~bash
git clone https://github.com/foodmind-team/foodmind-ml.git
cd foodmind-ml
uv sync --frozen --dev
uv run ruff format --check .
uv run ruff check .
uv run pytest -W error
uv pip check
~~~

Build the checked-in candidate artifact into the local runtime-package location used by FoodMind Intelligence:

~~~bash
uv run python scripts/build_runtime_package.py --output .tmp/runtime/model-package
~~~

This creates a local, generated package. Do not commit .tmp/.

## Local deployment and runtime handoff

This repository is an offline build and validation environment, not a public
service. Build the model package before running the Intelligence repository
independently; the integrated Infrastructure stack performs its own package
build from the pinned ML source.

On Windows PowerShell:

```powershell
git clone https://github.com/foodmind-team/foodmind-ml.git
Set-Location foodmind-ml
uv sync --frozen --dev
uv run python scripts/build_runtime_package.py --output .tmp/runtime/model-package
Get-ChildItem .tmp/runtime/model-package
```

The generated directory contains the validated manifest, model artifact, and
checksums consumed by the private Inference service. For a standalone
Intelligence checkout, keep this path at
`foodmind-ml/.tmp/runtime/model-package` so its diagnostic Compose volume mount
can read it. For normal end-to-end local development, use
[FoodMind Infrastructure](https://github.com/foodmind-team/foodmind-infra)
instead of publishing the package or opening an ML port.

Use `uv run pytest -W error` and `uv pip check` before handing a package to a
runtime. Remove generated local state only when it is no longer needed; `.tmp`
is ignored and must never contain production data, credentials, or a model
claimed to be production-approved without the release process.

## Configuration and credentials

ML has no runtime HTTP endpoint and no API-token configuration. Its normal
local setup is the locked `uv` environment plus file paths supplied to the
build scripts:

```powershell
uv run python scripts/build_runtime_package.py --output .tmp/runtime/model-package
# Only for an approved, access-controlled training snapshot:
uv run python scripts/build_collaborative_index.py `
  --snapshot <approved-local-snapshot.ndjson> `
  --output <local-collaborative-index.json>
```

Treat snapshot locations, approved-data access, and generated artifacts as
local security boundaries. Do not add database credentials, Backend service
tokens, provider API keys, or cloud access keys to this repository. Runtime
tokens belong to Infra/Backend/Intelligence; ML hands over only a validated
package with its manifest and checksums.

## Model handoff

~~~text
Approved snapshot --> validation/features --> train/evaluate --> package + checksums
                                                              |
                                                              '--> FoodMind Intelligence validates and serves it
~~~

A package includes a manifest, feature and inference-contract versions, source snapshot identity, evaluation summary, model-card reference, and SHA-256 checksums. The private inference service verifies it before loading.

Collaborative indexes require a time-bounded HMAC-pseudonymised Backend snapshot. Menu or catalogue seed data is not interaction data and must not enable UserCF or ItemCF.

~~~bash
uv run python scripts/build_collaborative_index.py --snapshot /secure/training-snapshot.ndjson --output /secure/collaborative-index.json
uv run python scripts/build_runtime_package.py --collaborative-index /secure/collaborative-index.json --output /secure/model-package
~~~

## Repository layout

~~~text
src/foodmind_ml/     Reusable feature and collaborative-index code
scripts/             Training, validation, packaging, and data-preparation tools
tests/               Deterministic unit and package-contract tests
artifacts/           Checked-in candidate artifacts and metadata
data/                Versioned non-sensitive inputs and processed data
reports/             Evaluation outputs
docs/                Experiment and model-card documentation
~~~

## Reproducibility and contribution

Record the source snapshot, configuration, random seed, training commit, metrics, and known limitations for every experiment. Add deterministic tests for transformations, cold starts, leakage checks, and package checksums. A candidate package is not an approved production model merely because the test suite passes.

## License

No open-source license is currently included in this repository. Obtain permission from the maintainers before redistributing or reusing the code.

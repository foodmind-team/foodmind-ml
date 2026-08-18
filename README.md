# FoodMind ML

FoodMind ML is FoodMind's offline model-development and packaging repository. It validates approved data snapshots, builds hybrid recommendation features, evaluates candidate models, and produces an immutable model package for FoodMind Intelligence.

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

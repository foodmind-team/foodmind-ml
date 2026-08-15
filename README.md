# FoodMind ML

FoodMind ML is the offline model-development repository for FoodMind. It prepares data, engineers features, calculates collaborative signals, trains and evaluates the Logistic Regression acceptance model, and publishes a versioned model package for `foodmind-intelligence`.

> **Runtime status:** the local package includes the trained hybrid artifact.
> UserCF and ItemCF are enabled only when an approved Backend training snapshot
> has been converted to a verified collaborative index. Menu/catalogue data is
> not user-interaction data and never enables collaborative filtering.

## Repository Role

This repository owns:

- Dataset definitions and snapshot metadata
- Data validation and preprocessing
- User-meal interaction-matrix construction
- UserCF and ItemCF research and feature generation
- Logistic Regression training
- Baseline and model evaluation
- Reproducible experiments
- Model cards and limitations
- Model-package construction
- Candidate and approved release metadata

This repository does not own:

- Runtime Agent graphs
- Runtime inference HTTP endpoints
- Public client or backend APIs
- User authentication or authorisation
- Production database access
- Online model retraining
- Android or Web code

Runtime loading and serving belong to `foodmind-intelligence`.

## Handoff to FoodMind Intelligence

```text
Data snapshot
  → validate
  → build features
  → train/evaluate
  → package immutable release
  → publish manifest + checksum
  → foodmind-intelligence validates and loads
```

The handoff is a versioned model-package contract, not copied source code.

## Repository Structure

```text
foodmind-ml/
├── data/
│   └── processed/
├── docs/
│   ├── experiments/
│   └── model-cards/
├── reports/
│   └── metrics/
├── scripts/
├── src/foodmind_ml/
│   ├── __init__.py
│   └── pipeline.py
└── tests/
    └── unit/
```
│   └── packaging/
└── tests/
    ├── unit/
    ├── integration/
    └── reproducibility/
```

## Model Design

FoodMind uses a feature-level hybrid recommendation design:

1. Spring Boot removes candidates that violate hard constraints.
2. UserCF produces a similar-user signal when interaction data is sufficient.
3. ItemCF produces a similar-meal signal when interaction data is sufficient.
4. Logistic Regression combines collaborative, preference, context, history, and group features.
5. Runtime code estimates acceptance probability.
6. Runtime returns a stable ordered candidate set with the lead candidate first.
7. The Recommendation Agent explains verified reason codes; the score is not the explanation.

Hard constraints must never be replaced by model predictions.

## Label Semantics

Training labels:

- Explicit acceptance: positive label
- Explicit rejection: negative label
- Passive non-selection: unknown, not negative

Collaborative interaction strength may also use:

- Later rating
- Would Eat Again
- Trusted-group evidence

The dataset must keep supervised labels distinct from collaborative interaction weights.

## Data Governance

- Do not commit production data, personal identifiers, credentials, or unrestricted exports.
- Raw and derived datasets require provenance and permitted-use records.
- Synthetic or manually labelled data must be clearly disclosed.
- Use stable anonymous identifiers for modelling.
- Retain dataset snapshot/checksum metadata for reproducibility.
- Remove future information from training features.
- Prefer a time-aware split for event data.
- Do not claim production accuracy from a small or simulated dataset.

Large data and model binaries should live in approved external storage, not Git history.

## Feature Contract

Candidate features may include:

- `user_cf_score`
- `user_cf_available`
- `item_cf_score`
- `item_cf_available`
- Cuisine preference match
- Spice compatibility
- Price-to-budget ratio
- Area or location match
- Meal-time match
- Days since Meal, cuisine, or Place
- Personal mean rating
- Trusted-group mean rating and interaction count
- Want to Try signal
- Novelty/exploratory indicator

The final feature list, types, defaults, and order must be versioned. Missing collaborative signals require availability flags and must not be treated as dislike.

## Evaluation Requirements

Report:

- Dataset origin and size
- Class balance
- Split method
- Leakage checks
- Baseline definition
- Confusion matrix
- Precision
- Recall
- F1
- ROC-AUC
- Calibration information when practical
- Cold-start behaviour
- Fallback rate
- Known limitations
- Top-1 lead-choice quality and top-3 ranking quality
- Personal, Exploratory, and Group-inspired segment coverage

Evaluation must be reproducible from a configuration and an immutable dataset snapshot.

The client may display one lead recommendation at a time, but offline
evaluation must assess the full ordered candidate set rather than only the
visible card.

## Model Package

An approved package should include:

- Manifest
- Serialized preprocessing/model artifacts
- Model version
- Training-code commit
- Dataset snapshot identifier
- Feature-schema version
- Inference-contract version
- Evaluation summary
- Model-card reference
- SHA-256 checksums

The package is consumed by the `foodmind-intelligence/inference-service`. See [model release process](docs/model-cards/model-release-process.md).

### Collaborative-index handoff

Export a time-bounded, HMAC-pseudonymised Backend snapshot, then build the
index outside the serving stack:

```powershell
$env:PYTHONPATH = 'src'
python scripts/build_collaborative_index.py `
  --snapshot D:\secure-export\training-snapshot.ndjson `
  --output D:\secure-release\collaborative-index.json
python scripts/build_runtime_package.py `
  --collaborative-index D:\secure-release\collaborative-index.json `
  --output D:\secure-release\model-package
```

The index is positive-only: accepted recommendations, ratings of at least four,
and `WOULD_EAT_AGAIN=true` can contribute; a rejection and passive
non-selection cannot. The runtime checks both the index checksum and its source
snapshot checksum metadata. Without the index, `userCf.available` and
`itemCf.available` remain `false`.

## Experiment Rules

- Notebooks are for exploration and communication.
- Reusable transformations move into `src/foodmind_ml`.
- Training results must be reproducible without manually executing notebook cells.
- Every result records configuration, seed, dataset snapshot, and Git commit.
- Candidate comparison uses the same evaluation split and metric definitions.
- A failed experiment does not overwrite a previous release.

## Configuration

Configuration files should separate:

- Data source and snapshot
- Feature schema
- Training parameters
- Evaluation parameters
- Packaging metadata

Secrets and local filesystem paths must not be committed in shared configuration.

## Testing Strategy

- Unit tests for feature transformations
- Toy-matrix tests for UserCF and ItemCF
- Missing/cold-start tests
- Dataset-schema and leakage tests
- Deterministic training smoke tests
- Package-manifest and checksum tests
- Reproducibility tests
- Consumer-contract fixtures shared with FoodMind Intelligence

## Contribution Workflow

1. Define the experiment question and success criteria.
2. Confirm data provenance and snapshot.
3. Add or update versioned configuration.
4. Implement reusable work outside notebooks.
5. Run unit, integration, and reproducibility tests.
6. Record metrics and limitations.
7. Create a candidate package only after evaluation.
8. Obtain review before marking a model release approved.

## Further Reading

- [Training workflow](docs/experiments/training-workflow.md)
- [Model release process](docs/model-cards/model-release-process.md)

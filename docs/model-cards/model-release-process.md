# Model Release Process

## Release States

```text
experiment
  → candidate
  → validated
  → approved
  → published
  → deployed by FoodMind Intelligence
  → monitored or rolled back
```

Only an approved and published immutable package may be used by staging or production-demo.

## Versioning

Use a unique model version that never changes after publication. The version should identify a release, not merely a filename such as `latest`.

Changes that require a new model version include:

- Trained coefficients
- Preprocessing state
- Collaborative artefacts
- Feature schema
- Dataset snapshot
- Training configuration that changes output

Contract compatibility is tracked separately through explicit schema versions.

## Required Manifest Fields

The model-package schema should require:

- `modelVersion`
- `createdAt`
- `trainingCommit`
- `datasetSnapshot`
- `datasetChecksum`
- `featureSchemaVersion`
- `inferenceContractVersion`
- `artifactFiles`
- Per-file SHA-256
- `evaluationReport`
- `modelCard`
- `supportedRuntime`
- `previousApprovedVersion`

## Release Gates

### Data gate

- Provenance documented
- Permitted use confirmed
- No unauthorised personal data
- Snapshot immutable
- Validation report passed

### Model gate

- Baseline comparison complete
- Evaluation metrics recorded
- Lead-candidate and ordered top-three behaviour recorded
- Personal, Exploratory, and Group-inspired coverage reviewed
- Cold-start behaviour tested
- Leakage checks passed
- Limitations reviewed

### Package gate

- Manifest schema valid
- Checksums valid
- Clean-environment loading passed
- Consumer fixtures passed
- Runtime contract compatible

### Review gate

- At least one reviewer
- Model card complete
- Rollback version identified
- Publishing destination approved

## Publishing

Large binary artifacts should be published to an approved model registry or object store. Git should contain:

- Contract schemas
- Small examples
- Manifests
- Metrics
- Model cards
- Release references

Git should not contain production data or large serialized models.

## Handoff Record

Provide FoodMind Intelligence:

```text
MODEL_VERSION=<immutable-version>
MODEL_ARTIFACT_URI=<immutable-uri>
MODEL_ARTIFACT_SHA256=<package-checksum>
FEATURE_SCHEMA_VERSION=<schema-version>
INFERENCE_CONTRACT_VERSION=<contract-version>
```

Supply values through an approved deployment channel, not a public README or committed secret file.

## Model Card Minimum Content

- Intended use
- Out-of-scope use
- Training data description
- Label semantics
- Feature groups
- Evaluation method and metrics
- Cold-start behaviour
- Fairness/representation limitations
- Privacy considerations
- Operational fallback
- Approval and version history

## Rollback Support

Every release identifies the previous approved version. Do not delete a predecessor required for rollback during the demonstration window.

Rollback does not modify this training repository's historical release. Runtime deployment selects a previous immutable package.

## Revocation

Revoke a release if:

- The package checksum is invalid.
- Data provenance is unacceptable.
- A serious leakage or privacy issue is discovered.
- Runtime outputs violate the feature/contract assumptions.
- The artifact cannot be reproduced or loaded safely.

A revoked version must not be overwritten or silently republished.

# Training Workflow

## Objective

Produce a reproducible, explainable acceptance-ranking model package that can be validated and consumed by FoodMind Intelligence.

## 1. Define the Experiment

Record:

- Question or hypothesis
- Candidate model/baseline
- Dataset snapshot
- Label definition
- Feature-schema version
- Split method
- Primary and secondary metrics
- Acceptance threshold
- Known constraints

An experiment without these fields is exploratory and cannot produce a release.

## 2. Acquire and Register Data

For each source, retain:

- Source name
- Ownership/licence or permitted use
- Acquisition date
- Snapshot identifier
- File/object checksums
- Row counts
- Schema version
- Whether data is real, anonymised, synthetic, or manually labelled

Never place unrestricted personal exports under `data/`.

## 3. Validate Data

Validate before feature generation:

- Required columns
- Identifier uniqueness
- Timestamp ordering and timezone
- Rating and price ranges
- Enum values
- Missing-value rate
- Duplicate events
- Label consistency
- Group/visibility fields excluded from unauthorised modelling inputs

Produce a validation report rather than silently dropping unexpected data.

## 4. Split Before Learning

For interaction events, prefer a time-aware split:

- Training contains earlier events.
- Validation/test contain later events.
- Transformations learn parameters from training only.
- User and item leakage is measured and reported.

Keep the final test set untouched during feature/model selection.

## 5. Build Interaction Semantics

Maintain separate structures for:

### Supervised label

- Accepted: `1`
- Explicitly rejected: `0`
- Not selected: missing/unknown

### Collaborative strength

May represent:

- Acceptance
- Rating
- Would Eat Again
- Explicit rejection

Document the exact mapping and sensitivity tests.

## 6. Generate Collaborative Features

UserCF:

- Build user-meal vectors.
- Calculate cosine similarity.
- Select neighbour policy.
- Produce score and availability flag.

ItemCF:

- Build meal interaction vectors.
- Calculate cosine similarity.
- Select similar-item policy.
- Produce score and availability flag.

Toy matrices must verify known results.

## 7. Build Candidate Features

Generate point-in-time-safe features such as:

- Preference match
- Price/budget compatibility
- Context match
- Recency and repetition
- Historical user rating
- Trusted-group evidence
- Want to Try
- UserCF and ItemCF

No feature may read data created after the prediction timestamp.

## 8. Establish Baselines

Compare against at least:

- Majority-class classifier
- Preference/context-only Logistic Regression
- Deterministic ranking used by Backend fallback

The hybrid model must be evaluated against a meaningful simpler approach.

## 9. Train and Evaluate

Record:

- Random seed
- Hyperparameters
- Library versions
- Train/validation/test counts
- Class balance
- Confusion matrix
- Precision, recall, F1, ROC-AUC
- Cold-start segments
- Calibration or probability distribution
- Error analysis
- Top-1 metrics for the lead recommendation
- Top-3 ranking/coverage metrics for the ordered candidate set
- Personal, Exploratory, and Group-inspired cohort coverage

Do not select a release using the final test set repeatedly.

The product spotlights one candidate initially, but the model contract still
supports up to three intentionally different candidates. Evaluation must
therefore report both lead-choice performance and ordered-set quality.

## 10. Review Explainability

Confirm:

- Feature names are stable and understandable.
- Model coefficients are inspectable.
- Runtime reason codes come from verified evidence, not free-form model interpretation.
- Sensitive or proxy features are reviewed.
- Model limitations are documented.

## 11. Package

Build a candidate model package containing the required manifest, artifacts, schemas, metrics, checksums, and model-card reference.

The packaging step must not depend on notebook state.

## 12. Consumer Validation

Use the fixtures shared with FoodMind Intelligence:

- Normal known user/item
- User cold start
- Item cold start
- Both collaborative features unavailable
- Boundary values
- Invalid feature schema

The candidate is not releasable until consumer validation passes.

## 13. Approval

Record:

- Reviewer
- Approved package version
- Metrics report
- Model card
- Compatibility versions
- Registry URI and checksum
- Rollback predecessor

Publishing a binary alone is not a release.

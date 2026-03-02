# Loan Default Risk Prediction Competition

A production-grade machine learning competition for evaluating AI agents on realistic tabular data prediction tasks.

## Overview

This repository contains a complete ML competition package for predicting consumer loan defaults. The challenge includes:

- **20,000 synthetic loan records** with realistic complexity
- **Binary classification task** with 7% positive class (class imbalance)
- **Grouped data structure** requiring proper validation strategy
- **High-cardinality categorical features** and structured missingness
- **Feature interactions** and realistic noise patterns
- **Deterministic scoring system** using AUC-ROC metric

This competition is designed to test genuine ML competence while resisting trivial solutions.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0

### 2. Generate Dataset

```bash
cd scripts
python generate_dataset.py
```

This creates five CSV files in the `data/` directory:
- `train.csv` - Training data with target variable (14,000 rows)
- `test.csv` - Test data without target (6,000 rows)
- `solution.csv` - Ground truth for test set (for scoring only)
- `sample_submission.csv` - Submission format template
- `perfect_submission.csv` - Perfect predictions for validation

### 3. Build Your Model

See [docs/instruction.md](docs/instruction.md) for detailed competition instructions.

See [docs/golden_workflow.md](docs/golden_workflow.md) for comprehensive ML reasoning and best practices.

### 4. Score Your Submission

```bash
python scripts/score_submission.py \
    --submission-path <your_submission.csv> \
    --solution-path data/solution.csv
```

Output: A single AUC-ROC score (0.0 to 1.0, higher is better)

### Test the Scoring Pipeline

```bash
# Test with perfect predictions (should output 1.0)
python scripts/score_submission.py \
    --submission-path data/perfect_submission.csv \
    --solution-path data/solution.csv

# Test with random predictions (should output ~0.5)
python scripts/score_submission.py \
    --submission-path data/sample_submission.csv \
    --solution-path data/solution.csv
```

## Repository Structure

```
.
├── data/
│   ├── train.csv                 # Training set with target
│   ├── test.csv                  # Test set without target
│   ├── solution.csv              # Ground truth for scoring
│   ├── sample_submission.csv     # Submission template
│   └── perfect_submission.csv    # Validation file
│
├── docs/
│   ├── dataset_card.md          # Comprehensive dataset documentation
│   ├── instruction.md           # Competition instructions
│   └── golden_workflow.md       # ML best practices and reasoning
│
├── scripts/
│   ├── generate_dataset.py      # Synthetic data generation
│   └── score_submission.py      # Deterministic scoring script
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Competition Details

### Task
Predict the probability of loan default for each loan in the test set.

### Metric
**AUC-ROC** (Area Under the Receiver Operating Characteristic Curve)
- Measures ranking quality of predictions
- Robust to class imbalance
- Threshold-independent
- Range: 0.0 (worst) to 1.0 (perfect)

### Key Challenges

1. **Grouped Data Structure**
   - ~2,000 borrowers have multiple loans
   - Requires GroupKFold validation by borrower_id
   - Naive random CV causes information leakage

2. **Class Imbalance**
   - Only 7% of loans default
   - Requires proper stratification and class weighting
   - Accuracy is a poor metric

3. **High-Cardinality Categoricals**
   - State (50 values), loan purpose (10 values)
   - Needs thoughtful encoding strategy
   - Target encoding must be done within CV to avoid leakage

4. **Structured Missingness**
   - Employment length (15% missing)
   - Delinquency history (60% missing)
   - Missingness itself is informative

5. **Feature Interactions**
   - Default risk depends on combinations of factors
   - Linear models underperform
   - Tree ensembles (LightGBM, XGBoost) recommended

### Expected Performance

| Approach | Expected AUC |
|----------|--------------|
| Random predictions | 0.50 |
| Logistic regression (basic) | 0.70-0.73 |
| Random Forest (default) | 0.74-0.77 |
| LightGBM (proper CV) | 0.78-0.80 |
| Tuned ensemble | 0.80-0.82 |

## Documentation

### [Dataset Card](docs/dataset_card.md)
Comprehensive documentation including:
- Data generation methodology
- Feature descriptions
- License information
- Leakage risks
- Ethical considerations
- Why the dataset is non-trivial

### [Competition Instructions](docs/instruction.md)
Complete instructions for participants:
- Dataset description
- Submission format requirements
- Evaluation metric explanation
- Important constraints
- Baseline performance ranges

### [Golden Workflow](docs/golden_workflow.md)
In-depth ML reasoning including:
- Data sanity checks
- Proper validation strategy (GroupKFold)
- Preprocessing pipeline
- Baseline models
- Iterative improvement strategy
- Error analysis techniques
- Robustness considerations

**This is the most important document** for understanding how to approach the competition correctly.

## Design Philosophy

This competition is built to:

- **Test real ML skills:** Requires proper validation, feature engineering, and modeling
- **Resist trivial solutions:** Grouped structure and interactions prevent naive approaches from succeeding
- **Expose leakage vulnerabilities:** Tests whether practitioners understand CV pitfalls
- **Reward principled methodology:** The gap between correct and incorrect validation is substantial
- **Maintain legal compliance:** Fully synthetic data with clean licensing
- **Support deterministic evaluation:** Reproducible scoring with no randomness

## Technical Specifications

### Dataset Properties
- **Size:** 20,000 loans (14,000 train / 6,000 test)
- **Features:** 17 predictive features + identifiers
- **Target:** Binary (0 = paid, 1 = default)
- **Default rate:** ~7%
- **Borrowers:** ~18,000 unique (~2,000 with multiple loans)
- **Split method:** Grouped by borrower_id, stratified by target

### Compliance
- **License:** MIT License (full ownership)
- **Data source:** 100% synthetic (generated from scratch)
- **Randomness:** Deterministic generation (seed=42)
- **Scoring:** Deterministic AUC-ROC calculation
- **No external data:** All dependencies are standard Python libraries

### Validation
- **Deterministic:** Same submission always produces same score
- **Format checking:** Validates columns, IDs, ranges, missing values
- **Security:** No code execution, only CSV parsing and scoring

## Usage Examples

### Basic Workflow

```python
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Prepare features
X = train.drop(columns=['default', 'loan_id', 'borrower_id', 'application_year_month'])
y = train['default']
groups = train['borrower_id']

# Basic preprocessing
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())

# Proper grouped validation
gkf = GroupKFold(n_splits=5)
cv_scores = []

for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    cv_scores.append(auc)

print(f"Mean CV AUC: {np.mean(cv_scores):.4f}")

# Train on full data and predict test set
model.fit(X, y)
X_test = test.drop(columns=['loan_id', 'borrower_id', 'application_year_month'])
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

test_pred = model.predict_proba(X_test)[:, 1]

# Create submission
submission = pd.DataFrame({
    'loan_id': test['loan_id'],
    'default': test_pred
})
submission.to_csv('my_submission.csv', index=False)
```

## Contributing

This is a research-grade ML benchmark. Contributions to improve documentation, fix bugs, or enhance the evaluation framework are welcome.

## License

MIT License

Copyright (c) 2026 Project Carrot ML Benchmarking Team

See [dataset_card.md](docs/dataset_card.md) for full license text.

## Citation

If you use this competition dataset in research or publication, please reference:

```
Loan Default Risk Prediction Competition (2026)
Project Carrot ML Benchmarking Initiative
https://github.com/your-org/kagel-project
```

## Contact

For questions, issues, or feedback, please open an issue in this repository or refer to the documentation.

---

**Built for rigorous ML evaluation. Designed to challenge both humans and AI agents.**

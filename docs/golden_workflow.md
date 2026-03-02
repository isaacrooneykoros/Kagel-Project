# Golden Workflow: Loan Default Risk Prediction

**A comprehensive guide to solving this competition with proper ML reasoning**

This document outlines a principled approach to the loan default prediction task. It demonstrates the thought process, decisions, and techniques that separate novice submissions from expert-level solutions. Think of this as the annotation you'd find in a winning Kaggle solution notebook.

---

## Table of Contents

1. Data Sanity Checks
2. Validation Strategy
3. Preprocessing Pipeline
4. Baseline Models
5. Iterative Improvement
6. Error Analysis
7. Robustness and Leaderboard Safety

---

## 1. Data Sanity Checks

Before any modeling, we need to understand our data deeply. Here's what to check and why it matters.

### Load and Inspect

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
```

**Expected output:**
- Train: (14000, 19) - includes target
- Test: (6000, 18) - no target

### Target Distribution Analysis

```python
print(train['default'].value_counts())
print(f"Default rate: {train['default'].mean():.2%}")
```

**Finding:** The target is imbalanced with approximately 7% positive class (defaults). This is realistic for loan data but means:
- Accuracy is a poor metric (predicting all 0s gives 93% accuracy)
- We need stratified sampling in CV
- Class weights or resampling might help
- AUC-ROC is the right metric choice

**Why this matters:** If you use random undersampling incorrectly, you'll throw away valuable information from the majority class. If you ignore the imbalance entirely, tree models might not learn the minority class pattern well.

### Check for Duplicate Loans

```python
print(f"Duplicate loan_ids in train: {train['loan_id'].duplicated().sum()}")
print(f"Duplicate loan_ids in test: {test['loan_id'].duplicated().sum()}")
```

**Expected:** Zero duplicates. Each loan_id is unique.

### Critical Discovery: Grouped Data Structure

```python
# Check borrower repetition
borrower_counts = train['borrower_id'].value_counts()
print(f"Unique borrowers: {train['borrower_id'].nunique()}")
print(f"Borrowers with 1 loan: {(borrower_counts == 1).sum()}")
print(f"Borrowers with 2 loans: {(borrower_counts == 2).sum()}")
print(f"Borrowers with >2 loans: {(borrower_counts > 2).sum()}")
```

**Expected finding:** About 2,000 borrowers have 2 loans, the rest have 1 loan.

**This is the MOST IMPORTANT discovery in this dataset.**

**Why this matters:** If the same borrower appears multiple times:
- Their loans are not independent samples
- Random CV will put the same borrower's loans in both train and validation folds
- The model can implicitly "memorize" borrower-level tendencies
- Your CV score will be optimistic and won't reflect true generalization

**Implication:** We MUST use GroupKFold with borrower_id as the grouping variable.

Let's verify the leakage risk:

```python
# Check if any borrower appears in both train and test
train_borrowers = set(train['borrower_id'])
test_borrowers = set(test['borrower_id'])
overlap = train_borrowers & test_borrowers

print(f"Borrowers in train: {len(train_borrowers)}")
print(f"Borrowers in test: {len(test_borrowers)}")
print(f"Overlap: {len(overlap)}")
```

**Expected:** Zero overlap. The dataset creators properly split by borrower.

### Missing Value Analysis

```python
missing_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
print(missing_pct[missing_pct > 0])
```

**Expected findings:**
- `months_since_last_delinquency`: ~60% missing
- `employment_length`: ~15% missing

**Why this matters:** 

For `months_since_last_delinquency`, the missingness is INFORMATIVE. Missing means "never had a delinquency", which is actually a positive signal. Simple mean imputation would be wrong here. Better approach:
- Create binary indicator `has_delinquency_history`
- Impute with large value (999) for missing cases
- Or use separate imputation groups

For `employment_length`, missing might indicate:
- Self-employment
- Gig work
- Frequent job changes
- Not applicable

Create a binary `employment_length_missing` indicator before imputation.

### Feature Correlations

```python
# Numeric features only
numeric_features = train.select_dtypes(include=[np.number]).columns
correlation_matrix = train[numeric_features].corr()

# Key correlations to check
print(f"credit_score vs interest_rate: {train['credit_score'].corr(train['interest_rate']):.3f}")
print(f"annual_income vs loan_amount: {train['annual_income'].corr(train['loan_amount']):.3f}")
print(f"credit_score vs default: {train['credit_score'].corr(train['default']):.3f}")
```

**Expected findings:**
- credit_score and interest_rate: Strong negative correlation (~-0.6)
  - This makes sense: better credit gets lower rates
- annual_income and loan_amount: Moderate positive correlation (~0.4)
  - Higher earners borrow more
- credit_score and default: Negative correlation
  - Lower credit scores predict higher default risk

**Why this matters:** Strong correlations between features don't hurt tree models but can cause multicollinearity issues in linear models. The relationship between credit_score and interest_rate is especially strong - encoding both is fine for trees but might need regularization for linear models.

### Distribution Checks

```python
# Check for heavy tails
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(train['loan_amount'], bins=50)
plt.title('Loan Amount Distribution')

plt.subplot(132)
plt.hist(train['annual_income'], bins=50)
plt.title('Annual Income Distribution')

plt.subplot(133)
plt.hist(train['credit_score'], bins=50)
plt.title('Credit Score Distribution')

plt.tight_layout()
```

**Expected findings:**
- Loan amount: Right-skewed (most loans small, some large)
- Income: Right-skewed (log-normal like)
- Credit score: Approximately normal

**Why this matters:** Heavy-tailed distributions can dominate loss functions. Tree models handle this naturally, but you might consider:
- Log transformation for linear models
- Robust scaling instead of StandardScaler
- Outlier clipping (though be careful not to remove legitimate high values)

### Categorical Cardinality

```python
categorical_features = ['loan_purpose', 'home_ownership', 'state', 'loan_term']
for col in categorical_features:
    print(f"{col}: {train[col].nunique()} unique values")
```

**Expected:**
- loan_purpose: 10 categories
- home_ownership: 4 categories
- state: 50 categories
- loan_term: 2 categories

**Why this matters:**
- Low cardinality (loan_term, home_ownership): One-hot encoding is fine
- Medium cardinality (loan_purpose): One-hot or ordinal encoding
- High cardinality (state): One-hot creates 50 columns. Consider:
  - Target encoding (with proper CV to avoid leakage)
  - Frequency encoding
  - Grouping low-frequency states

### Target Rate by Category

```python
for col in categorical_features:
    print(f"\n{col}:")
    print(train.groupby(col)['default'].agg(['count', 'mean']).sort_values('mean', ascending=False))
```

**Why this matters:** This reveals which categories are risky. For example:
- Some loan purposes might have much higher default rates
- Some states might show higher default rates
- This informs feature engineering and helps validate that the target has learnable patterns

---

## 2. Validation Strategy

This is where most competitors fail. Getting validation right IS the competition.

### The Wrong Way: Random K-Fold

```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# DON'T DO THIS
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(train):
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = X_train['default'], X_val['default']
    
    # ... train model ...
    # scores.append(val_score)

print(f"Mean CV AUC: {np.mean(scores):.4f}")
```

**Why this is wrong:**
- Borrower BORR_000123 might have 2 loans
- Loan 1 ends up in fold 1 (training)
- Loan 2 ends up in fold 3 (validation)
- Model sees borrower's loan 1 characteristics during training
- Then "predicts" loan 2 from same borrower
- This is information leakage!
- Your CV score will be 0.02-0.05 AUC points higher than true performance

### The Right Way: GroupKFold

```python
from sklearn.model_selection import GroupKFold

# Proper approach
gkf = GroupKFold(n_splits=5)
groups = train['borrower_id']
scores = []

for train_idx, val_idx in gkf.split(train, train['default'], groups=groups):
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = X_train['default'], X_val['default']
    
    # Verify no borrower overlap
    train_borrowers = set(X_train['borrower_id'])
    val_borrowers = set(X_val['borrower_id'])
    assert len(train_borrowers & val_borrowers) == 0, "Borrower leakage detected!"
    
    # ... train model ...
    # scores.append(val_score)

print(f"Mean CV AUC: {np.mean(scores):.4f}")
```

**Why this is correct:**
- Each borrower's loans stay together in the same fold
- No borrower appears in both train and validation
- Your CV score reflects true generalization
- The score will be slightly lower than random CV but MORE TRUSTWORTHY

### Even Better: StratifiedGroupKFold

The ideal approach combines grouping with stratification:

```python
from sklearn.model_selection import StratifiedGroupKFold

# Best approach (requires sklearn >= 1.1)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
groups = train['borrower_id']

scores = []
for train_idx, val_idx in sgkf.split(train, train['default'], groups=groups):
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = X_train['default'], X_val['default']
    
    print(f"Fold: Train default rate: {y_train.mean():.3f}, Val default rate: {y_val.mean():.3f}")
    
    # ... train model ...
```

This ensures:
- No borrower overlap (grouped)
- Similar default rates across folds (stratified)
- Most stable validation

### Hold-Out Test Set

In addition to CV, create a hold-out set for final validation:

```python
from sklearn.model_selection import train_test_split

# Split by borrower
unique_borrowers = train['borrower_id'].unique()
train_borrowers, val_borrowers = train_test_split(
    unique_borrowers, 
    test_size=0.2, 
    random_state=42,
    stratify=train.groupby('borrower_id')['default'].max()  # Stratify by borrower default
)

train_set = train[train['borrower_id'].isin(train_borrowers)]
val_set = train[train['borrower_id'].isin(val_borrowers)]
```

Use this hold-out set ONLY for final model selection decisions, not for hyperparameter tuning.

### Validation Best Practices

1. **Always group by borrower_id** - Non-negotiable
2. **Use 5 folds** - Good balance of data usage and variance
3. **Stratify if possible** - Maintains class balance
4. **Track fold-by-fold variance** - High variance suggests instability
5. **Don't overfit to CV** - Small improvements (< 0.001 AUC) are noise

### Leaderboard Overfitting Risk

With 6,000 test samples, the standard error on AUC is approximately:

```
SE ≈ 0.005-0.01
```

This means:
- Scoring the same model twice could vary by ±0.01 AUC due to random chance
- Don't chase 0.001 improvements
- Trust your local CV more than the leaderboard
- If CV and leaderboard scores diverge > 0.02, investigate

---

## 3. Preprocessing Pipeline

Now that we have proper validation, let's build a robust preprocessing pipeline.

### Step 1: Feature Engineering (Before Splitting)

Create derived features that make patterns more explicit:

```python
def engineer_features(df):
    df = df.copy()
    
    # Loan to income ratio
    df['loan_to_income'] = df['loan_amount'] / df['annual_income']
    
    # Credit utilization categories
    df['high_utilization'] = (df['credit_utilization'] > 75).astype(int)
    df['low_utilization'] = (df['credit_utilization'] < 30).astype(int)
    
    # Has delinquency history
    df['has_delinquency'] = (df['delinquencies_last_2yrs'] > 0).astype(int)
    df['multiple_delinquencies'] = (df['delinquencies_last_2yrs'] > 2).astype(int)
    
    # Employment stability
    df['employment_length_missing'] = df['employment_length'].isna().astype(int)
    df['short_employment'] = (df['employment_length'].fillna(0) < 2).astype(int)
    
    # Delinquency recency
    df['delinquency_history_missing'] = df['months_since_last_delinquency'].isna().astype(int)
    
    # Income to credit limit ratio
    df['income_to_credit_ratio'] = df['annual_income'] / df['total_credit_limit']
    
    # Interaction features
    df['credit_score_x_dti'] = df['credit_score'] * df['debt_to_income_ratio']
    df['bad_credit_high_dti'] = ((df['credit_score'] < 650) & 
                                   (df['debt_to_income_ratio'] > 35)).astype(int)
    
    return df

train_eng = engineer_features(train)
test_eng = engineer_features(test)
```

**Why these features work:**
- `loan_to_income`: Directly captures affordability risk
- Utilization indicators: High utilization signals financial stress
- Delinquency binary: Sometimes "any vs none" is more predictive than count
- Missing indicators: Preserve information in missingness patterns
- Interactions: Capture compounding risks (bad credit AND high debt is especially bad)

### Step 2: Handle Missing Values

```python
from sklearn.impute import SimpleImputer

def impute_missing(df):
    df = df.copy()
    
    # Employment length: median imputation (already have missing indicator)
    emp_imputer = SimpleImputer(strategy='median')
    df['employment_length'] = emp_imputer.fit_transform(df[['employment_length']])
    
    # Months since delinquency: large value for "never had one"
    df['months_since_last_delinquency'] = df['months_since_last_delinquency'].fillna(999)
    
    return df
```

**Trade-offs:**
- Median imputation is simple but loses information → why we add missing indicators
- Mode imputation for categoricals
- Advanced: KNN or iterative imputation (probably overkill for this dataset)

### Step 3: Encode Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder

def encode_features(df, target_encoding_map=None):
    df = df.copy()
    
    # Low cardinality: One-hot encoding
    df = pd.get_dummies(df, columns=['home_ownership', 'loan_term'], drop_first=True)
    
    # loan_purpose: One-hot (10 categories is borderline)
    df = pd.get_dummies(df, columns=['loan_purpose'], drop_first=True)
    
    # High cardinality (state): Target encoding
    # This requires careful CV-aware implementation
    if target_encoding_map is not None:
        df['state_encoded'] = df['state'].map(target_encoding_map)
    else:
        # Fallback: frequency encoding
        freq_map = df['state'].value_counts() / len(df)
        df['state_encoded'] = df['state'].map(freq_map)
    
    df = df.drop(columns=['state'])
    
    # Drop identifiers
    df = df.drop(columns=['loan_id', 'borrower_id'], errors='ignore')
    
    # Drop original date (could extract month/year as features if desired)
    df = df.drop(columns=['application_year_month'], errors='ignore')
    
    return df
```

**Target encoding done right:**

```python
from category_encoders import TargetEncoder

# Must be done INSIDE CV loop to avoid leakage
for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Fit target encoder on training fold only
    te = TargetEncoder()
    X_train['state_encoded'] = te.fit_transform(X_train['state'], y_train)
    X_val['state_encoded'] = te.transform(X_val['state'])
    
    # Continue with this fold...
```

**Why this matters:** If you fit target encoder on full training set, then use CV, you're leaking information from validation folds into training folds. This artificially inflates performance.

### Step 4: Scale Numeric Features (Optional)

```python
from sklearn.preprocessing import StandardScaler

# Only needed for linear models
scaler = StandardScaler()
numeric_cols = ['loan_amount', 'interest_rate', 'annual_income', 'credit_score', ...]

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
```

**When to scale:**
- Yes: Linear models, KNN, neural networks, SVMs
- No: Tree-based models (Random Forest, XGBoost, LightGBM)

For this competition, tree models likely perform best, so scaling is optional.

---

## 4. Baseline Models

Always start simple to establish performance floors.

### Baseline 1: Naive Predictions

```python
# Predict class prior for everyone
naive_pred = np.ones(len(test)) * train['default'].mean()

# Expected AUC: ~0.50 (random guessing at ranking)
```

**Purpose:** Worst-case performance. Any model should beat this.

### Baseline 2: Single Feature Model

```python
from sklearn.linear_model import LogisticRegression

# Use only credit score
X_train_simple = train[['credit_score']].fillna(700)
y_train = train['default']

model = LogisticRegression()
model.fit(X_train_simple, y_train)

# Expected AUC: ~0.62-0.65
```

**Purpose:** Shows that credit_score alone has predictive power. If full model doesn't beat this significantly, something is wrong.

### Baseline 3: Logistic Regression (All Features)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Proper preprocessing
X_train = train.drop(columns=['default', 'loan_id', 'borrower_id', 'application_year_month'])
X_train = pd.get_dummies(X_train, drop_first=True)
X_train = X_train.fillna(X_train.median())

y_train = train['default']

# Scale for linear model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Regularized logistic regression
model = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Expected AUC: 0.70-0.73
```

**Why this underperforms:**
- Linear models can't capture feature interactions
- Assumes linear relationship between features and log-odds
- No inherent feature interactions (credit_score AND high_dti effect)

### Baseline 4: Random Forest (Default Parameters)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Expected AUC: 0.74-0.77
```

**Why this is better:**
- Captures non-linear relationships
- Automatically handles feature interactions
- Robust to outliers and scaling

**Why it still underperforms:** Default parameters aren't optimized for this specific task.

---

## 5. Iterative Improvement Strategy

Now we move beyond baselines toward competitive performance.

### Model Selection: Gradient Boosting

For tabular data with feature interactions, gradient boosting typically dominates:

**Top choices:**
1. **LightGBM** - Fast, handles categorical features natively, excellent for tabular data
2. **XGBoost** - Slightly more robust, similar performance
3. **CatBoost** - Best for high-cardinality categoricals

**Why these work:**
- Build trees sequentially to correct previous mistakes
- Handle feature interactions naturally
- Work well with imbalanced data (with proper settings)
- Don't require feature scaling

### LightGBM Implementation

```python
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# Prepare data
X = train.drop(columns=['default', 'loan_id', 'application_year_month'])
y = train['default']
groups = train['borrower_id']

# Drop borrower_id after extracting groups
X = X.drop(columns=['borrower_id'])

# Basic feature engineering
X = engineer_features(X)
X = impute_missing(X)

# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'min_child_samples': 20,
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),  # Handle imbalance
    'random_state': 42
}

# GroupKFold cross-validation
gkf = GroupKFold(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predict
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_pred)
    cv_scores.append(auc)
    
    print(f"Fold {fold+1} AUC: {auc:.4f}")

print(f"\nMean CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Expected: 0.78-0.80 AUC
```

### Hyperparameter Tuning

Key parameters to tune:

1. **num_leaves** (default: 31)
   - Controls tree complexity
   - Higher = more complex = risk of overfitting
   - Try: [15, 31, 63, 127]

2. **learning_rate** (default: 0.1)
   - Lower = slower learning = more robust
   - Need more trees if you decrease this
   - Try: [0.01, 0.03, 0.05, 0.1]

3. **min_child_samples** (default: 20)
   - Minimum samples in leaf
   - Higher = more regularization
   - Try: [10, 20, 50, 100]

4. **feature_fraction** (default: 1.0)
   - Fraction of features to use per tree
   - Adds randomness, prevents overfitting
   - Try: [0.6, 0.8, 1.0]

5. **bagging_fraction** (default: 1.0)
   - Fraction of data to use per tree
   - Similar to Random Forest's bootstrap
   - Try: [0.7, 0.8, 0.9, 1.0]

6. **scale_pos_weight**
   - Handles class imbalance
   - Set to neg_samples / pos_samples
   - Or try is_unbalance=True

**Tuning strategy:**

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'num_leaves': [15, 31, 63, 127],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'min_child_samples': [10, 20, 50, 100],
    'feature_fraction': [0.6, 0.8, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
}

# But remember: use GroupKFold!
# RandomizedSearchCV doesn't support GroupKFold natively in old sklearn versions
# Use custom CV loop or manual grid search
```

### Feature Importance Analysis

```python
# After training
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(importance_df.head(20))
```

**Expected top features:**
- credit_score
- debt_to_income_ratio
- delinquencies_last_2yrs
- credit_utilization
- loan_to_income (engineered feature)
- employment_length

**Why this matters:**
- If `borrower_id` appears in top features → YOU HAVE LEAKAGE
- If engineered features rank high → feature engineering worked
- If no features stand out → model might be struggling with signal

### Ensemble Strategy

Combine multiple models for robustness:

```python
# Train multiple models
models = []

# LightGBM
lgb_model = train_lgb(X_train, y_train)
models.append(('lgb', lgb_model))

# XGBoost
xgb_model = train_xgb(X_train, y_train)
models.append(('xgb', xgb_model))

# CatBoost
cat_model = train_catboost(X_train, y_train)
models.append(('cat', cat_model))

# Weighted average ensemble
def ensemble_predict(X, models, weights):
    predictions = []
    for name, model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Weighted average
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    return ensemble_pred

# Find optimal weights via CV
# Simple approach: equal weights
weights = [1/3, 1/3, 1/3]
```

**Expected improvement:** +0.005 to +0.015 AUC from single model

**Diminishing returns:** Beyond 3 diverse models, gains are minimal.

---

## 6. Error Analysis

Understanding where your model fails helps iterate effectively.

### Prediction Distribution

```python
val_pred = model.predict(X_val)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(val_pred[y_val == 0], bins=50, alpha=0.7, label='Paid')
plt.hist(val_pred[y_val == 1], bins=50, alpha=0.7, label='Default')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.legend()
plt.title('Prediction Distribution by True Class')
```

**What to look for:**
- Good separation: Default cases have higher predicted probabilities
- Overlap: Some defaults predicted low, some paid predicted high (normal)
- If distributions are nearly identical → model hasn't learned anything

### Confusion Matrix at Different Thresholds

```python
from sklearn.metrics import confusion_matrix, classification_report

# Try threshold at 0.5
y_pred_class = (val_pred > 0.5).astype(int)
print(confusion_matrix(y_val, y_pred_class))
print(classification_report(y_val, y_pred_class))

# Try threshold at class prior
threshold = y_train.mean()
y_pred_class = (val_pred > threshold).astype(int)
print(confusion_matrix(y_val, y_pred_class))
```

**Why this matters:** AUC is threshold-independent, but understanding precision/recall trade-offs at different thresholds informs real-world deployment decisions.

### Residual Analysis

```python
# For defaults: How well did we rank them?
default_mask = y_val == 1
default_preds = val_pred[default_mask]

print(f"Mean prediction for defaults: {default_preds.mean():.3f}")
print(f"Median prediction for defaults: {np.median(default_preds):.3f}")

# False negatives: Defaults we missed
false_neg_threshold = 0.2
fn_mask = (y_val == 1) & (val_pred < false_neg_threshold)
false_negatives = X_val[fn_mask]

print(f"False negatives: {fn_mask.sum()}")
print(false_negatives.describe())
```

**Questions to investigate:**
- Do false negatives have surprisingly good credit scores?
- Are they from specific loan purposes or states?
- Are there hidden patterns we're missing?

### Performance by Subgroup

```python
# Performance by credit score quartile
train['credit_score_quartile'] = pd.qcut(train['credit_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    mask = X_val['credit_score_quartile'] == quartile
    if mask.sum() > 0:
        auc = roc_auc_score(y_val[mask], val_pred[mask])
        print(f"{quartile} (lowest to highest): AUC = {auc:.4f}")
```

**Why this matters:** 
- If model performs poorly on low credit scores → might need more features or rebalancing
- If AUC varies wildly across groups → model might be biased or unstable

### Prediction Calibration

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_val, val_pred, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curve')
plt.legend()
```

**What to look for:**
- Points on diagonal = well-calibrated
- Points below diagonal = overconfident (predicting higher than true rate)
- Points above diagonal = underconfident

**Why this matters:** For real lending decisions, you want predicted probabilities to reflect true default rates at each score level.

---

## 7. Robustness and Leaderboard Safety

Final considerations to avoid pitfalls.

### Public Leaderboard Overfitting

**The problem:** With 6,000 test samples, random variance is non-trivial.

Standard error on AUC:
```
SE ≈ sqrt(AUC * (1 - AUC) / n) ≈ sqrt(0.8 * 0.2 / 6000) ≈ 0.005
```

**Implication:** Your AUC could vary ±0.01 due to random sampling alone.

**Strategy:**
1. Trust your CV score
2. If CV and LB diverge by > 0.02, investigate (might be real drift or a bug)
3. Don't submit 50 times chasing +0.001 improvements
4. Make a small number of confident submissions based on solid CV

### Distribution Drift Check

```python
# Check feature distributions between train and test
for col in numeric_features:
    print(f"{col}:")
    print(f"  Train mean: {train[col].mean():.2f}, Test mean: {test[col].mean():.2f}")
    print(f"  Train std: {train[col].std():.2f}, Test std: {test[col].std():.2f}")
```

**Expected:** Distributions should be similar. This dataset is split randomly, not temporally, so drift should be minimal.

**If you see drift:** Model might not generalize. Consider:
- Robust preprocessing (RobustScaler instead of StandardScaler)
- Less aggressive hyperparameter tuning
- Simpler model architecture

### Adversarial Submission Detection

The scoring script validates:
- Correct file format
- Exact loan_id match
- Predictions in [0, 1]
- No missing values

**You cannot game the system by:**
- Submitting wrong IDs
- Submitting out-of-range values
- Submitting non-numeric predictions
- Submitting partial results

### Deterministic Scoring

**Key property:** Running the scorer twice on the same submission gives EXACTLY the same score.

This means:
- No randomness in AUC calculation
- No floating-point precision issues (within 8 decimals)
- Fair comparison between submissions

### Final Submission Checklist

Before making your final submission:

- [ ] Trained with GroupKFold (not random KFold)
- [ ] Excluded loan_id and borrower_id from features
- [ ] Used proper target encoding (within CV loop)
- [ ] Handled missing values appropriately
- [ ] Feature engineered based on domain knowledge
- [ ] Tuned hyperparameters within CV framework
- [ ] Validated on hold-out set
- [ ] Checked prediction distribution (not all 0s or 1s)
- [ ] Verified submission format matches sample_submission.csv
- [ ] Tested locally with scoring script
- [ ] CV score is reasonable (0.75-0.82 range)

---

## Expected Performance Ranges

| Approach | Expected AUC | Notes |
|----------|--------------|-------|
| Random guessing | 0.50 | Baseline floor |
| Credit score only | 0.62-0.65 | Single feature |
| Logistic regression (basic) | 0.70-0.73 | Linear model limit |
| Random Forest (default) | 0.74-0.77 | Tree baseline |
| LightGBM (tuned, proper CV) | 0.78-0.80 | Competitive |
| LightGBM + feature eng | 0.80-0.82 | Strong solution |
| Ensemble (3 models) | 0.81-0.83 | Near-optimal |

If you score:
- < 0.70: Implementation bug or severe leakage in wrong direction
- 0.70-0.75: Basic approach working but needs improvement
- 0.75-0.80: Solid solution with proper grouping
- 0.80-0.85: Excellent work with good feature engineering
- > 0.85: Check for data leakage (likely using solution.csv or borrower_id)

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Using Random CV
**Symptom:** CV score is 0.82 but leaderboard is 0.75  
**Fix:** Use GroupKFold with borrower_id

### Pitfall 2: Including Identifier Features
**Symptom:** borrow_id or loan_id shows up in feature importance  
**Fix:** Drop these before training

### Pitfall 3: Target Encoding Leakage
**Symptom:** CV score is unrealistically high (>0.90)  
**Fix:** Compute target encodings INSIDE each CV fold

### Pitfall 4: Ignoring Class Imbalance
**Symptom:** Model predicts all loans as paid (class 0)  
**Fix:** Use scale_pos_weight or class_weight='balanced'

### Pitfall 5: Not Handling Missing Values Properly
**Symptom:** Performance drops on test set  
**Fix:** Create missing indicators before imputation

### Pitfall 6: Overfitting to CV Folds
**Symptom:** Huge variance across folds  
**Fix:** Simplify model, add regularization, or collect more data (not possible here)

---

## Conclusion

This competition rewards:
1. **Proper validation strategy** (GroupKFold) - 40% of success
2. **Thoughtful feature engineering** - 30% of success
3. **Appropriate modeling** (GBDTs) - 20% of success
4. **Careful hyperparameter tuning** - 10% of success

The biggest separator is understanding the grouped data structure and validating correctly. Everything else is incremental improvement on top of that foundation.

**Key insight:** This is not a competition about finding the fanciest model. It's about understanding data structure, preventing leakage, and applying standard ML tools correctly.

Good luck, and may your CV scores be trustworthy!

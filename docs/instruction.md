# Competition Instructions: Loan Default Risk Prediction

## Objective

Your task is to predict whether borrowers will default on their loans. Given historical loan application data and outcomes, build a model that predicts the probability of default for loans in the test set.

This is a **binary classification** problem where you must predict a probability value between 0 and 1 for each loan.

## Dataset Description

### Files Provided

You have access to the following files in the `data/` directory:

1. **train.csv** - Training data with known outcomes
   - Contains 14,000 loan records
   - Includes all features and the target variable `default`
   - Use this data to train your model

2. **test.csv** - Test data for predictions
   - Contains 6,000 loan records
   - Includes all features EXCEPT the target variable
   - Make predictions for these loans

3. **sample_submission.csv** - Example submission format
   - Shows the correct format for your predictions
   - Contains `loan_id` and `default` columns
   - All predictions are set to 0.5 as a placeholder

4. **solution.csv** - Ground truth (DO NOT USE for training)
   - Contains the true default values for the test set
   - Used only by the scoring script to evaluate your submission
   - Using this file for training is prohibited and defeats the purpose of the competition

### Data Fields

Each loan record contains the following information:

- **loan_id**: Unique identifier for the loan
- **borrower_id**: Identifier for the borrower (some borrowers have multiple loans)
- **loan_amount**: Amount borrowed in dollars
- **interest_rate**: Annual interest rate as a percentage
- **loan_term**: Length of loan in months (36 or 60)
- **loan_purpose**: Reason for the loan
- **annual_income**: Borrower's yearly income in dollars
- **debt_to_income_ratio**: Monthly debt payments as percentage of monthly income
- **credit_score**: FICO credit score (580-850)
- **employment_length**: Years at current job (may be missing)
- **home_ownership**: Housing status (RENT, MORTGAGE, OWN, OTHER)
- **state**: US state where borrower resides
- **num_credit_lines**: Number of open credit accounts
- **credit_utilization**: Percentage of available credit being used
- **delinquencies_last_2yrs**: Count of late payments in past 2 years
- **months_since_last_delinquency**: Time since last late payment (may be missing)
- **total_credit_limit**: Total credit available across all accounts
- **application_year_month**: When the loan was applied for (YYYY-MM format)

**Target Variable:**
- **default**: Binary indicator (0 = loan paid off, 1 = loan defaulted)
  - Only present in train.csv
  - Your goal is to predict this for test.csv

## Submission Format

Your submission must be a CSV file with exactly two columns:

1. **loan_id** - The loan identifier from test.csv
2. **default** - Your predicted probability of default (must be between 0.0 and 1.0)

### Requirements

Your submission file must meet these requirements:

- **File format**: CSV (comma-separated values)
- **Columns**: Exactly two columns named `loan_id` and `default`
- **Column order**: Either order is acceptable
- **Row count**: Must contain exactly 6,000 rows (one for each test loan)
- **Loan IDs**: Must match exactly with the loan IDs in test.csv
- **Predictions**: 
  - Must be numeric values
  - Must be in the range [0.0, 1.0] inclusive
  - Represent probability of default (higher = more likely to default)
  - No missing values allowed
- **Header row**: Must include column names

### Example Format

```csv
loan_id,default
LOAN_000001,0.234
LOAN_000002,0.891
LOAN_000003,0.045
...
```

The order of rows does not matter as long as all loan IDs are present.

## Evaluation Metric

Submissions are evaluated using **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve).

### What is AUC-ROC?

AUC-ROC measures how well your model ranks predictions. It evaluates whether loans that actually defaulted receive higher predicted probabilities than loans that were paid off.

- **Range**: 0.0 to 1.0
- **Higher is better**
- **Random predictions**: ~0.50 AUC
- **Perfect predictions**: 1.00 AUC

AUC-ROC is robust to class imbalance and doesn't require choosing a classification threshold. It evaluates the quality of your probability estimates across all possible thresholds.

### Why AUC-ROC?

This metric is standard in credit risk modeling because:
- It handles imbalanced classes well (only 7% of loans default)
- It evaluates ranking quality, not just binary accuracy
- It's threshold-independent, focusing on probability calibration
- It aligns with real-world lending decisions (risk-based pricing)

## Scoring Your Submission

To evaluate your submission locally, use the provided scoring script:

```bash
python scripts/score_submission.py \
    --submission-path <your_submission.csv> \
    --solution-path data/solution.csv
```

The script will:
1. Validate your submission format
2. Check that all required loan IDs are present
3. Verify predictions are in valid range
4. Compute and print the AUC-ROC score

**Output**: A single number representing your AUC-ROC score (e.g., `0.75234891`)

**Exit codes**:
- 0 = Success
- 1 = Validation error (check error message)

### Testing Your Workflow

Before making real predictions, test your pipeline:

```bash
# Test with perfect predictions (should give 1.0)
python scripts/score_submission.py \
    --submission-path data/perfect_submission.csv \
    --solution-path data/solution.csv

# Test with random predictions (should give ~0.5)
python scripts/score_submission.py \
    --submission-path data/sample_submission.csv \
    --solution-path data/solution.csv
```

## Important Constraints

### What You MUST Do

1. **Use proper cross-validation**: The dataset contains borrowers with multiple loans. Use grouped validation (GroupKFold) to prevent leakage.

2. **Exclude certain features**: Do NOT use `loan_id` or `borrower_id` as predictive features. These are identifiers only.

3. **Handle missing values**: Some features have missing data (employment_length, months_since_last_delinquency). Handle these appropriately.

4. **Respect the train/test split**: Only use train.csv for training. Do not peek at solution.csv or use test.csv targets.

### What You SHOULD Avoid

1. **Naive cross-validation**: Random K-fold splits will overestimate your performance due to repeat borrowers. This will cause your local CV score to be optimistic compared to the true test score.

2. **Ignoring class imbalance**: Only 7% of loans default. Consider stratified sampling, class weights, or other techniques to handle imbalance.

3. **Overfitting**: With proper grouped validation, resist the temptation to overtune to small score improvements. Trust your CV process.

4. **Using identifiers as features**: loan_id and borrower_id carry no predictive information and will cause leakage issues.

## Baseline Performance

To help calibrate expectations, here are approximate performance levels:

- **Random predictions**: 0.50 AUC
- **Always predict mean**: 0.50 AUC  
- **Simple logistic regression**: 0.70-0.73 AUC
- **Random forest (default)**: 0.74-0.77 AUC
- **Well-tuned gradient boosting**: 0.78-0.82 AUC

If your model achieves below 0.65 AUC, there's likely an implementation issue. If you achieve above 0.85 AUC, double-check for data leakage.

## Recommended Workflow

1. **Exploratory Data Analysis**
   - Load and inspect train.csv
   - Check target distribution (class imbalance)
   - Identify repeat borrowers using borrower_id
   - Examine missing value patterns
   - Visualize feature distributions

2. **Validation Strategy**
   - Set up GroupKFold with borrower_id as groups
   - Use 5 folds for robust estimates
   - Stratify by target if possible
   - Verify no borrower appears in multiple folds

3. **Preprocessing**
   - Handle missing values (imputation + indicators)
   - Encode categorical variables (one-hot or target encoding)
   - Scale numeric features if using linear models
   - Consider feature engineering (ratios, interactions)

4. **Modeling**
   - Start with simple baseline (logistic regression)
   - Try tree-based models (Random Forest, XGBoost, LightGBM)
   - Tune hyperparameters within CV framework
   - Consider ensembling multiple models

5. **Evaluation**
   - Validate using your GroupKFold setup
   - Check predictions are well-calibrated
   - Perform error analysis on misclassified cases
   - Generate final predictions on test set

6. **Submission**
   - Create submission file in correct format
   - Validate using the scoring script
   - Compare test score to CV score (should be similar)

## Getting Help

If you encounter issues:

1. **Format errors**: Read the error message from the scoring script carefully. It will tell you what's wrong with your submission format.

2. **Unexpected scores**: If your test score is much different from CV score, you likely have a validation strategy problem. Ensure you're using GroupKFold.

3. **Low performance**: Review the golden workflow documentation for detailed modeling guidance.

4. **Technical issues**: Check that you have the required packages installed (see requirements.txt).

## Additional Resources

- **dataset_card.md**: Detailed information about data generation, features, and leakage risks
- **golden_workflow.md**: Deep dive into proper modeling approach with ML reasoning
- **requirements.txt**: Python package dependencies

## Competition Rules Summary

- **Goal**: Predict loan default probabilities for test set
- **Metric**: AUC-ROC (higher is better)
- **Format**: CSV with loan_id and default columns
- **Validation**: Use GroupKFold by borrower_id
- **Constraint**: Do not use solution.csv for training
- **Success criteria**: Achieve >0.75 AUC with proper methodology

Good luck! This competition is designed to test real ML competence, so expect to need thoughtful feature engineering, proper validation, and appropriate modeling choices.

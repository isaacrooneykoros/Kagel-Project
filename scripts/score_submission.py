"""
Deterministic AUC-ROC scoring for loan default competition.

Usage: python score_submission.py --submission-path <path> --solution-path <path>
Outputs float AUC score to stdout
"""

import argparse
import sys
import pandas as pd
from pathlib import Path


def load_csv_safely(filepath, file_description):
    """Load CSV with error handling."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: {file_description} file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error: Failed to read {file_description}: {e}", file=sys.stderr)
        sys.exit(1)


def validate_columns(df, expected_cols, file_description):
    """Check that dataframe has expected columns."""
    actual_cols = set(df.columns)
    expected_set = set(expected_cols)
    
    if actual_cols != expected_set:
        missing = expected_set - actual_cols
        extra = actual_cols - expected_set
        
        msg = f"Error: {file_description} has incorrect columns.\n"
        if missing:
            msg += f"  Missing: {missing}\n"
        if extra:
            msg += f"  Unexpected: {extra}\n"
        msg += f"  Expected: {expected_cols}"
        
        print(msg, file=sys.stderr)
        sys.exit(1)


def validate_ids_match(submission_ids, solution_ids):
    """Verify that submission has exactly the same loan IDs as solution."""
    sub_set = set(submission_ids)
    sol_set = set(solution_ids)
    
    if sub_set != sol_set:
        missing = sol_set - sub_set
        extra = sub_set - sol_set
        
        msg = "Error: Submission loan_id values don't match solution.\n"
        if missing:
            msg += f"  Missing {len(missing)} IDs (first 5): {list(missing)[:5]}\n"
        if extra:
            msg += f"  Extra {len(extra)} IDs (first 5): {list(extra)[:5]}"
        
        print(msg, file=sys.stderr)
        sys.exit(1)


def validate_predictions(preds):
    """Validate predictions are in [0, 1]."""
    if preds.isna().any():
        n = preds.isna().sum()
        print(f"Error: {n} missing values", file=sys.stderr)
        sys.exit(1)
    
    try:
        preds = pd.to_numeric(preds)
    except (ValueError, TypeError):
        print("Error: Non-numeric predictions", file=sys.stderr)
        sys.exit(1)
    
    if (preds < 0).any() or (preds > 1).any():
        n_bad = ((preds < 0) | (preds > 1)).sum()
        print(f"Error: {n_bad} predictions outside [0, 1]", file=sys.stderr)
        sys.exit(1)
    
    return preds


def compute_auc_roc(y_true, y_pred):
    """Calculate AUC-ROC deterministically."""
    order = y_pred.argsort()[::-1]
    y_sorted = y_true[order]
    
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        print("Error: All same class", file=sys.stderr)
        sys.exit(1)
    
    tp = 0
    fp = 0
    auc = 0.0
    
    for val in y_sorted:
        if val == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    
    return auc / (n_pos * n_neg)


def main():
    """Main scoring function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Score loan default prediction submission'
    )
    parser.add_argument(
        '--submission-path',
        required=True,
        help='Path to submission CSV file'
    )
    parser.add_argument(
        '--solution-path',
        required=True,
        help='Path to solution CSV file'
    )
    
    args = parser.parse_args()
    
    # Load files
    submission = load_csv_safely(args.submission_path, "Submission")
    solution = load_csv_safely(args.solution_path, "Solution")
    
    # Validate columns
    expected_cols = ['loan_id', 'default']
    validate_columns(submission, expected_cols, "Submission")
    validate_columns(solution, expected_cols, "Solution")
    
    # Validate row counts match
    if len(submission) != len(solution):
        print(f"Error: Submission has {len(submission)} rows but solution has {len(solution)} rows", 
              file=sys.stderr)
        sys.exit(1)
    
    # Validate loan IDs match
    validate_ids_match(submission['loan_id'], solution['loan_id'])
    
    # Validate predictions
    validate_predictions(submission['default'])
    
    # Align submission to solution order by loan_id
    # This ensures we compare the right predictions to right labels
    submission_sorted = submission.set_index('loan_id').loc[solution['loan_id']].reset_index()
    
    # Extract arrays for scoring
    y_true = solution['default'].values
    y_pred = submission_sorted['default'].values
    
    # Compute AUC-ROC
    auc_score = compute_auc_roc(y_true, y_pred)
    
    # Output single float with 8 decimal precision
    print(f"{auc_score:.8f}")
    
    sys.exit(0)


if __name__ == '__main__':
    main()

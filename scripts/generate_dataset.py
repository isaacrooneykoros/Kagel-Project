#!/usr/bin/env python3
"""Generate synthetic loan default dataset for competition."""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NUM_LOANS = 20000
NUM_BORROWERS = 18000
TRAIN_RATIO = 0.7
DEFAULT_RATE = 0.07

US_STATES = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
             'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
             'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
             'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
             'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

HIGH_RISK_STATES = ['NV', 'FL', 'AZ', 'MI', 'CA']
LOAN_PURPOSES = ['debt_consolidation', 'credit_card', 'home_improvement',
                 'small_business', 'major_purchase', 'medical', 'car',
                 'moving', 'vacation', 'other']


def generate_borrower_ids(n_loans, n_borrowers):
    """Create borrower IDs with ~2k repeat borrowers."""
    repeat_count = 2000
    repeat_ids = list(range(repeat_count)) * 2
    single_ids = list(range(repeat_count, n_borrowers))
    all_ids = repeat_ids + single_ids
    np.random.shuffle(all_ids)
    return [f'BORR_{i:06d}' for i in all_ids[:n_loans]]


def generate_base_features(n):
    """Generate loan and borrower features."""
    data = {}
    data['loan_id'] = [f'LOAN_{i:06d}' for i in range(n)]
    data['borrower_id'] = generate_borrower_ids(n, NUM_BORROWERS)
    
    amt = np.random.lognormal(mean=9.5, sigma=0.5, size=n)
    data['loan_amount'] = np.clip(amt, 1000, 40000).astype(int)
    data['loan_term'] = np.random.choice([36, 60], size=n, p=[0.7, 0.3])
    
    probs = [0.25, 0.20, 0.15, 0.08, 0.08, 0.06, 0.06, 0.04, 0.04, 0.04]
    data['loan_purpose'] = np.random.choice(LOAN_PURPOSES, size=n, p=probs)
    
    cs = np.random.normal(loc=700, scale=60, size=n)
    data['credit_score'] = np.clip(cs, 580, 850).astype(int)
    
    inc = np.random.lognormal(mean=10.8, sigma=0.5, size=n)
    data['annual_income'] = np.clip(inc, 20000, 200000).astype(int)
    
    cs_norm = (data['credit_score'] - 580) / (850 - 580)
    rate = 28 - (cs_norm * 23)
    rate += np.random.normal(0, 1.5, n)
    data['interest_rate'] = np.clip(rate, 5.0, 28.0).round(2)
    
    mth_inc = data['annual_income'] / 12
    debt_pmt = data['loan_amount'] * 0.03
    other_debt = np.random.uniform(0, 0.25, n) * mth_inc
    dti = ((debt_pmt + other_debt) / mth_inc * 100).clip(0, 45)
    data['debt_to_income_ratio'] = dti.round(2)
    
    emp = np.random.exponential(scale=5, size=n).clip(0, 40)
    emp[np.random.random(n) < 0.15] = np.nan
    data['employment_length'] = emp
    
    home = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    data['home_ownership'] = np.random.choice(home, size=n, p=[0.45, 0.40, 0.12, 0.03])
    data['state'] = np.random.choice(US_STATES, size=n)
    
    data['num_credit_lines'] = np.clip(np.random.poisson(lam=8, size=n) + 2, 2, 30)
    
    inc_f = data['annual_income'] / 50000
    lim = inc_f * 30000 * np.random.uniform(0.7, 1.3, n)
    data['total_credit_limit'] = np.clip(lim.astype(int), 5000, 150000)
    
    data['credit_utilization'] = (np.random.beta(2, 5, n) * 100).round(2)
    
    has_del = np.random.random(n) > 0.80
    del_cnt = np.random.poisson(lam=1.5, size=n)
    data['delinquencies_last_2yrs'] = np.where(has_del, del_cnt, 0).clip(0, 8)
    
    mths = np.random.uniform(1, 120, n)
    mths[np.random.random(n) < 0.60] = np.nan
    data['months_since_last_delinquency'] = mths
    
    start = pd.Timestamp('2023-01-01')
    days = np.random.randint(0, 365 * 3, n)
    dates = [start + pd.Timedelta(days=int(d)) for d in days]
    data['application_year_month'] = [d.strftime('%Y-%m') for d in dates]
    
    return pd.DataFrame(data)


def generate_default_target(df):
    """Generate defaults with risk interactions."""
    n = len(df)
    risk = np.ones(n) * 0.07
    
    cs_fac = (850 - df['credit_score']) / 270
    risk += cs_fac * 0.15
    
    high_dti = (df['debt_to_income_ratio'] > 35).astype(float)
    risk += high_dti * 0.12
    
    bad_cs = (df['credit_score'] < 650).astype(float)
    risk += (bad_cs * high_dti) * 0.25
    
    loan_inc = df['loan_amount'] / df['annual_income']
    high_ratio = (loan_inc > 0.3).astype(float)
    risk += high_ratio * 0.15
    
    mult_del = (df['delinquencies_last_2yrs'] > 2).astype(float)
    risk += mult_del * 0.20
    
    short_emp = df['employment_length'].fillna(0) < 2
    unstable = (short_emp & (df['home_ownership'] == 'RENT')).astype(float)
    risk += unstable * 0.10
    
    hi_risk = df['state'].isin(HIGH_RISK_STATES).astype(float)
    risk += hi_risk * 0.08
    
    high_u = (df['credit_utilization'] > 75).astype(float)
    risk += high_u * 0.08
    
    risk += np.random.normal(0, 0.05, n)
    
    risk_scaled = (risk - 0.30) * 8
    prob = 1 / (1 + np.exp(-risk_scaled))
    
    default = (np.random.random(n) < prob).astype(int)
    
    current = default.mean()
    if abs(current - DEFAULT_RATE) > 0.01:
        diff = DEFAULT_RATE - current
        n_flip = int(abs(diff) * n)
        if diff > 0:
            idx = np.where(default == 0)[0]
            flip = np.random.choice(idx, n_flip, replace=False)
            default[flip] = 1
        else:
            idx = np.where(default == 1)[0]
            flip = np.random.choice(idx, n_flip, replace=False)
            default[flip] = 0
    
    return default


def split_train_test(df, ratio=0.7):
    """Split by borrower to prevent leakage."""
    bg = df.groupby('borrower_id')
    b_rate = bg['default'].mean()
    has_default = (b_rate > 0).astype(int)
    
    unique_b = b_rate.index.values
    np.random.seed(RANDOM_SEED)
    
    train_b = []
    test_b = []
    
    for has_d in [0, 1]:
        subset = unique_b[has_default == has_d]
        n_tr = int(len(subset) * ratio)
        shuffled = subset.copy()
        np.random.shuffle(shuffled)
        train_b.extend(shuffled[:n_tr])
        test_b.extend(shuffled[n_tr:])
    
    train = df[df['borrower_id'].isin(train_b)].reset_index(drop=True)
    test = df[df['borrower_id'].isin(test_b)].reset_index(drop=True)
    
    return train, test


def main():
    print("="*60)
    print("Generating Synthetic Loan Dataset")
    print("="*60)
    
    print(f"\nGenerating {NUM_LOANS:,} loans...")
    df = generate_base_features(NUM_LOANS)
    
    print("Computing defaults...")
    df['default'] = generate_default_target(df)
    
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Unique borrowers: {df['borrower_id'].nunique():,}")
    
    lc = df['borrower_id'].value_counts()
    multi = (lc > 1).sum()
    print(f"Repeat borrowers: {multi:,}\n")
    
    print("Splitting train/test by borrower...")
    train, test = split_train_test(df, TRAIN_RATIO)
    
    print(f"Train: {len(train):,} ({train['borrower_id'].nunique():,} borrowers)")
    print(f"Test:  {len(test):,} ({test['borrower_id'].nunique():,} borrowers)")
    print(f"Train default: {train['default'].mean():.2%}")
    print(f"Test default:  {test['default'].mean():.2%}\n")
    
    out = Path(__file__).parent.parent / 'data'
    out.mkdir(exist_ok=True)
    
    train.to_csv(out / 'train.csv', index=False)
    test.drop(columns=['default']).to_csv(out / 'test.csv', index=False)
    
    sol = test[['loan_id', 'default']]
    sol.to_csv(out / 'solution.csv', index=False)
    
    sample = sol.copy()
    sample['default'] = 0.5
    sample.to_csv(out / 'sample_submission.csv', index=False)
    sol.to_csv(out / 'perfect_submission.csv', index=False)
    
    print(f"Saved to {out}/")


if __name__ == '__main__':
    main()
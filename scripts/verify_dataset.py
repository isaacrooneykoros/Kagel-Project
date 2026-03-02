import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sol = pd.read_csv('data/solution.csv')

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)
print()

print("Train:")
print(f"  Shape: {train.shape}")
print(f"  Default rate: {train['default'].mean():.2%}")
print(f"  Unique borrowers: {train['borrower_id'].nunique()}")

lc = train['borrower_id'].value_counts()
print(f"  Single-loan: {(lc == 1).sum()}")
print(f"  Repeat: {(lc > 1).sum()}\n")

print("Test:")
print(f"  Shape: {test.shape}")
print(f"  Default rate: {sol['default'].mean():.2%}")
print(f"  Unique borrowers: {test['borrower_id'].nunique()}\n")

train_b = set(train['borrower_id'])
test_b = set(test['borrower_id'])
overlap = train_b & test_b
print(f"Overlap: {len(overlap)}")
if len(overlap) == 0:
    print("  [OK] No leakage")
else:
    print("  [ERROR] Leakage detected!")

# Check missing values
print("Missing values in train:")
missing = train.isnull().sum()
missing = missing[missing > 0]
for col, count in missing.items():
    pct = count / len(train) * 100
    print(f"  {col}: {count} ({pct:.1f}%)")
print()

print("=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)

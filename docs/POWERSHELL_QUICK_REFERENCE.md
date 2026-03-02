# PowerShell Quick Reference for Competition

## Running Python Scripts in PowerShell

### Important: Line Continuation Syntax
---

## Command Examples

### 1. Generate Dataset

```powershell
python scripts/generate_dataset.py
```

**Output:** Creates all CSV files in `data/` directory

---

### 2. Score a Submission

**Option A: All on one line**
```powershell
python scripts/score_submission.py --submission-path data/sample_submission.csv --solution-path data/solution.csv
```

**Option B: Multi-line (with backtick continuation)**
```powershell
python scripts/score_submission.py `
    --submission-path data/sample_submission.csv `
    --solution-path data/solution.csv
```

**Output:** Single AUC-ROC score (e.g., `0.47878274`)

---

### 3. Verify Dataset Structure

```powershell
python scripts/verify_dataset.py
```

**Output:** Dataset statistics and verification checks

---

## Example Results

### Random Predictions (sample_submission.csv)
```
0.47878274
```

### Perfect Predictions (perfect_submission.csv)
```
1.00000000
```

---

## Key Syntax Rules for PowerShell

| Situation | Syntax | Example |
|-----------|--------|---------|
| **Line continuation** | Backtick `` ` `` | `command ` ` + next line` |
| **Single line** | All on one | `command --arg1 value1 --arg2 value2` |
| **Relative paths** | Forward or back slash | `data/file.csv` or `data\file.csv` |
| **Backslash in paths** | Windows paths work | `C:\Users\Admin\file.csv` |

---

## Troubleshooting

### Error: "Missing expression after unary operator"
**Cause:** Using `\` instead of `` ` `` for line continuation  
**Fix:** Replace `\` with `` ` ``

### Error: "Python command not found"
**Cause:** Python not in PATH  
**Fix:** Use full path: `c:/python313/python.exe scripts/...`

### Error: "submission.csv not found"
**Cause:** Wrong path or file doesn't exist  
**Fix:** Use full path: `data/submission.csv` or `./data/submission.csv`

---

## Complete Workflow

```powershell
# Step 1: Generate dataset
python scripts/generate_dataset.py

# Step 2: Create your submission (outside of PowerShell, in your Python notebook or script)

# Step 3: Score your submission
python scripts/score_submission.py --submission-path my_submission.csv --solution-path data/solution.csv

# Step 4: Check dataset structure (optional)
python scripts/verify_dataset.py
```

---

## File Locations

All commands assume you're running from: `C:\Users\Admin\PycharmProjects\Kagel-Project`

If you're in a different directory, adjust paths:

```powershell
# From project root
python scripts/generate_dataset.py

# From outside project
c:/python313/python.exe C:\Users\Admin\PycharmProjects\Kagel-Project\scripts\generate_dataset.py
```

---

## More Information

- **README.md** - Project overview
- **docs/instruction.md** - Competition rules
- **docs/golden_workflow.md** - ML approach guide
- **COMPLIANCE_REVIEW.md** - Technical details

---

**Status:** [PASS] Dataset Generated and Scoring System Validated  
**Date:** March 2, 2026

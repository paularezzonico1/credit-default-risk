# Credit Default Risk Predictor

An end-to-end machine learning project predicting borrower default on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrowers). Built to demonstrate the core data science and ML concepts relevant to financial risk teams.

---

## Results

| Model | Test ROC-AUC |
|---|---|
| Decision Tree (deep) | 0.613 |
| Decision Tree (shallow) | 0.818 |
| Logistic Regression (L1/L2) | 0.801 |
| Random Forest | 0.844 |
| **XGBoost** | **0.863** |

**Final model:** XGBoost at threshold 0.20, optimized for recall to reflect lending cost asymmetry (a missed default costs far more than a false alarm).

---

## Project Structure

```
credit-default-risk/
├── data/
│   └── cs-training.csv          # Raw data (not tracked by git)
├── notebooks/
│   └── Credit Default Risk Analysis.ipynb  # Full analysis: EDA → modeling → evaluation
├── executive_summary.png        # Output chart for stakeholders
└── README.md
```

---

## Methodology

### Phase 1 — EDA & Statistics
- Explored 150K borrowers across 11 features
- Handled missing data: `MonthlyIncome` (20% missing) and `NumberOfDependents` (2.6%) imputed with median — chosen for robustness to the heavily right-skewed income distribution
- Ran t-tests comparing defaulters vs. non-defaulters: age (p≈0), late payments (p≈0), and monthly income (p≈0) are all statistically significant; revolving utilization is not (p=0.49)

### Phase 2 — Unsupervised Segmentation
- Applied K-Means (k=4, chosen via elbow method) on standardized financial features
- Discovered a high-risk cluster (Cluster 2): 54.6% default rate, average 98 late payments, youngest and lowest-income borrowers
- Cluster label used as an engineered feature in downstream models

### Phase 3 — Baseline with Regularization
- Trained Logistic Regression with L1 (Lasso) and L2 (Ridge) penalties
- Both achieved ROC-AUC 0.80; coefficients confirmed past delinquency as the dominant signal
- L1 and L2 produced near-identical results, indicating no truly irrelevant features in this dataset

### Phase 4 — Model Progression & Bias-Variance
- Deep decision tree: Train AUC 1.0, Test AUC 0.61 — textbook overfitting
- Shallow tree: Train 0.81, Test 0.82 — good generalization, low variance
- Random Forest: ensembling recovered test performance (0.844) despite full-depth trees
- XGBoost: best generalization (0.863), smallest train/test gap

### Phase 5 — Validation Done Right
- Used a **time-based split** (80% train, 20% test by row order) instead of random split
- Prevents data leakage: random splits would allow future loan information to leak into training, inflating performance estimates — a critical issue in lending data

### Phase 6 — Evaluation & Threshold Tuning
- Accuracy is misleading on this 93/7 imbalanced dataset — reported ROC-AUC, precision, recall, F1
- Default threshold (0.50) yields only 21% recall on defaulters — unacceptable in a lending context
- Lowered threshold to **0.20**: recall rises to 50%, reflecting the business reality that a missed default (entire loan lost) far outweighs the cost of a false alarm (lost interest revenue)

---

## Key Interview Talking Points

1. **Leakage prevention:** Time-based split mirrors real-world deployment — you train on past loans and predict future ones. A random split would let future data leak into training.

2. **Why not accuracy?** A model predicting "no default" every time would be 93% accurate but completely useless. ROC-AUC and recall are the right metrics here.

3. **Threshold as a business decision:** The optimal threshold is not 0.50 — it depends on the cost asymmetry of the problem. In lending, I set it at 0.20 to prioritize catching defaulters.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/paularezzonico1/credit-default-risk.git

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

# Download data from Kaggle
# https://www.kaggle.com/c/GiveMeSomeCredit/data
# Place cs-training.csv in data/

# Launch notebook
jupyter notebook "notebooks/Credit Default Risk Analysis.ipynb"
```

---

## Tech Stack
`Python` `pandas` `scikit-learn` `XGBoost` `matplotlib` `seaborn` `scipy` `Jupyter`

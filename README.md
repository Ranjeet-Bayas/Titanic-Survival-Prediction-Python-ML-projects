# # 🚢 Titanic Survival Prediction – Machine Learning Classification

A machine learning project that predicts whether a Titanic passenger survived, using scikit-learn pipelines combining preprocessing and classification models. Two classifiers — **Random Forest** and **Logistic Regression** — are trained, tuned with GridSearchCV, and compared on accuracy and feature importance.

---

## 📌 Project Objectives

- Build a classification model using scikit-learn to predict passenger survival
- Implement an end-to-end **Pipeline** combining preprocessing and a classifier
- Handle missing values and mixed data types with a `ColumnTransformer`
- Tune hyperparameters using **GridSearchCV** with **Stratified K-Fold** cross-validation
- Interpret model results via classification report, confusion matrix, and feature importances
- Compare performance of **Random Forest** vs **Logistic Regression**

---

## 📊 Dataset

- **Source:** Seaborn built-in Titanic dataset (`sns.load_dataset('titanic')`)
- **Total records:** 891 passengers
- **Target variable:** `survived` (0 = Did not survive, 1 = Survived)
- **Class distribution:** 549 did not survive (62%) · 342 survived (38%) — slight imbalance, handled via stratified splits

### Features Used

| Feature | Type | Description |
|---|---|---|
| `pclass` | Numeric | Ticket class (1st, 2nd, 3rd) |
| `sex` | Categorical | Passenger sex |
| `age` | Numeric | Age in years (has missing values) |
| `sibsp` | Numeric | No. of siblings/spouses aboard |
| `parch` | Numeric | No. of parents/children aboard |
| `fare` | Numeric | Passenger fare |
| `class` | Categorical | Ticket class as text (First/Second/Third) |
| `who` | Categorical | Man, woman, or child |
| `adult_male` | Boolean | Whether passenger is an adult male |
| `alone` | Boolean | Whether passenger was travelling alone |

> **Dropped columns:** `deck` (77% missing), `embarked`, `embark_town`, `alive` (leakage risk)

---

## 🔧 Methodology

### 1. Preprocessing Pipeline

Separate pipelines were built for numerical and categorical features and merged using a `ColumnTransformer`:

**Numerical features** (`pclass`, `age`, `sibsp`, `parch`, `fare`):
- `SimpleImputer` — median strategy (handles missing `age` values)
- `StandardScaler` — standardise to zero mean and unit variance

**Categorical features** (`sex`, `class`, `who`, `adult_male`, `alone`):
- `SimpleImputer` — most-frequent strategy
- `OneHotEncoder` — encode categories (unknown values ignored)

### 2. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
- 80/20 split — stratified to preserve class ratio

### 3. Cross-Validation

```python
cv = StratifiedKFold(n_splits=5, shuffle=True)
```

### 4. Model 1 — Random Forest Classifier

**Hyperparameter grid searched:**

| Parameter | Values Tried |
|---|---|
| `n_estimators` | 50, 100 |
| `max_depth` | None, 10, 20 |
| `min_samples_split` | 2, 5 |

### 5. Model 2 — Logistic Regression

**Hyperparameter grid searched:**

| Parameter | Values Tried |
|---|---|
| `solver` | liblinear |
| `penalty` | l1, l2 |
| `class_weight` | None, balanced |

---

## 📈 Results

### Model Comparison

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| **Test Accuracy** | **81.56%** | **82.68%** |
| Precision (class 0) | 0.83 | 0.84 |
| Recall (class 0) | 0.87 | 0.89 |
| F1-score (class 0) | 0.85 | 0.86 |
| Precision (class 1) | 0.78 | 0.81 |
| Recall (class 1) | 0.72 | 0.72 |
| F1-score (class 1) | 0.75 | 0.76 |
| Macro Avg F1 | 0.80 | 0.81 |
| Weighted Avg F1 | 0.81 | 0.82 |

> ✅ **Logistic Regression slightly outperformed Random Forest** across all metrics, though the difference is marginal. Both models were evaluated on the same held-out test set (179 samples).

### Key Insight on Feature Importance

The two models assigned very different importance weights to features, despite similar accuracy. This highlights significant **multicollinearity** among features — for example, `who_man`, `who_woman`, and `who_child` are perfectly dependent (knowing two implies the third). A more rigorous correlation analysis is recommended before drawing conclusions about individual feature contributions.

---

## 🗂️ Project Structure

```
titanic-survival-prediction/
│
├── Titanic_survival_project.ipynb   # Main Jupyter notebook
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

Python 3.8+ is required. Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
# Clone the repository
git clone https://github.com/Ranjeet-Bayas/Data-science-projects.git
cd Data-science-projects

# Launch Jupyter
jupyter notebook "Titanic_survival_project (1).ipynb"
```

Or open directly in **Google Colab** — the notebook was originally developed there and runs without any additional setup.

---

## 📦 Requirements

```
numpy>=1.23
pandas>=1.2
matplotlib>=3.4
seaborn>=0.13
scikit-learn>=1.6
```

---

## 🛠️ Libraries & Tools

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` / `seaborn` | Data visualisation and confusion matrix heatmap |
| `scikit-learn` | Preprocessing, pipelines, models, GridSearchCV |

---

## 📝 Key Skills Demonstrated

- End-to-end ML pipeline with `sklearn.pipeline.Pipeline`
- Mixed-type feature preprocessing with `ColumnTransformer`
- Missing value imputation with `SimpleImputer`
- Hyperparameter tuning with `GridSearchCV`
- Stratified cross-validation with `StratifiedKFold`
- Model evaluation: classification report, confusion matrix, feature importances
- Comparison of tree-based vs linear classification models

---

## 👤 Author

**Ranjeet Bayas**
- GitHub: [Ranjeet-Bayas](https://github.com/Ranjeet-Bayas)

*Original lab template by Jeff Grossman (IBM Corporation)*

---

## 📄 License

This project is for educational purposes. The Titanic dataset is publicly available via the Seaborn library.

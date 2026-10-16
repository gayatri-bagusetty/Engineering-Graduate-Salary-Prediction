
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

# ============================================================
# 1. LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Engineering_graduate_salary.csv"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place Engineering_graduate_salary.csv beside this script."
    )

nb = pd.read_csv(DATA_PATH)

if "Salary" not in nb.columns:
    raise ValueError("Dataset must contain a Salary column.")

nb["Salary"] = pd.to_numeric(nb["Salary"], errors="coerce")
nb.dropna(subset=["Salary"], inplace=True)

print("\nFirst five rows:")
print(nb.head())

print("\nDataset shape:", nb.shape)
print("\nColumn names:")
print(nb.columns.tolist())

print("\nDataset information:")
nb.info()

print("\nMissing values:")
print(nb.isnull().sum())

print("\nStatistical summary:")
print(nb.describe(include="all").T)


# ============================================================
# 2. DATA QUALITY DIAGNOSTICS
# ============================================================

missing_data = pd.DataFrame({
    "Total": nb.isnull().sum(),
    "Percent": nb.isnull().mean() * 100
})

print("\nMissing Data Summary:")
print(missing_data.sort_values("Total", ascending=False))

unique_data = pd.DataFrame({
    "Column": nb.columns,
    "Unique Values": [nb[col].nunique() for col in nb.columns],
    "Data Type": [str(nb[col].dtype) for col in nb.columns]
})

print("\nUnique Values and Data Types:")
print(unique_data.to_string(index=False))

print("\n--- Salary distribution ---")
print(
    nb["Salary"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\nHighest 15 salaries:")
print(nb["Salary"].nlargest(15).to_list())

print("\nSalary skewness:", round(nb["Salary"].skew(), 4))

if "GraduationYear" in nb.columns:
    print("\nGraduationYear values:")
    print(nb["GraduationYear"].value_counts().sort_index())


# ============================================================
# 3. CLEAN DATA
# ============================================================

# Remove identifier columns that are unlikely to generalize.
columns_to_drop = ["ID", "CollegeID", "CollegeCityID", "DOB"]
nb.drop(columns=columns_to_drop, errors="ignore", inplace=True)

# Graduation year 0 is treated as unknown.
if "GraduationYear" in nb.columns:
    nb.loc[nb["GraduationYear"] == 0, "GraduationYear"] = np.nan

# Scores equal to -1 appear to indicate unavailable or
# non-applicable specialization scores.
score_columns = [
    "ComputerProgramming",
    "ElectronicsAndSemicon",
    "ComputerScience",
    "MechanicalEngg",
    "ElectricalEngg",
    "TelecomEngg",
    "CivilEngg"
]

print("\n--- Engineering scores equal to -1 ---")

for col in score_columns:
    if col in nb.columns:
        count = (nb[col] == -1).sum()
        print(f"{col}: {count} records ({count / len(nb):.1%})")

        # Preserve information about whether the score was unavailable.
        nb[f"{col}_unavailable"] = (nb[col] == -1).astype("int8")

        # The numeric imputer will fill these missing scores later.
        nb.loc[nb[col] == -1, col] = np.nan

# Clean Specialization values.
if "Specialization" in nb.columns:
    nb["Specialization"] = (
        nb["Specialization"].astype("string").str.strip()
    )

    nb["Specialization"] = nb["Specialization"].str.replace(
        "electronics & instrumentation eng",
        "electronics and instrumentation engineering",
        case=False,
        regex=False
    )

    # Combine specializations with 10 or fewer records.
    counts = nb["Specialization"].value_counts()
    rare_categories = counts[counts <= 10].index

    nb.loc[
        nb["Specialization"].isin(rare_categories),
        "Specialization"
    ] = "other"

print("\nDataset after cleaning:")
print(nb.head())
print("\nCleaned dataset shape:", nb.shape)


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / filename,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


# Missing values
plt.figure(figsize=(12, 5))
sns.heatmap(nb.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values After Cleaning")
save_plot("missing_values.png")


# 10th vs 12th percentage
if {"10percentage", "12percentage"}.issubset(nb.columns):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=nb,
        x="10percentage",
        y="12percentage"
    )
    plt.title("10th Percentage vs 12th Percentage")
    save_plot("percentage_scatter.png")


# Personality features and salary
personality_cols = [
    col for col in [
        "conscientiousness",
        "agreeableness",
        "extraversion",
        "nueroticism",
        "openess_to_experience",
        "Salary"
    ]
    if col in nb.columns
]

if len(personality_cols) >= 2:
    pair_data = nb[personality_cols].dropna()

    if len(pair_data) > 1000:
        pair_data = pair_data.sample(1000, random_state=101)

    pair_plot = sns.pairplot(pair_data, corner=True)
    pair_plot.fig.suptitle(
        "Personality Features and Salary",
        y=1.02
    )
    pair_plot.savefig(
        PLOT_DIR / "personality_pairplot.png",
        dpi=120
    )
    plt.close("all")


# Correlation heatmap
numeric_data = nb.select_dtypes(include=np.number)
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
save_plot("correlation_heatmap.png")


# Numeric histograms
numeric_data.hist(
    figsize=(16, 12),
    bins=30,
    edgecolor="black"
)
plt.suptitle("Numeric Feature Distributions", y=1.02)
save_plot("histograms.png")


# Salary distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(nb["Salary"], kde=True)
plt.title("Salary Distribution")

plt.subplot(1, 2, 2)
sorted_salary = np.sort(nb["Salary"].to_numpy())
plt.scatter(range(len(sorted_salary)), sorted_salary, s=10)
plt.title("Sorted Salary Curve")
plt.xlabel("Sorted Record")
plt.ylabel("Salary")

save_plot("salary_distribution.png")


# Gender distribution
if "Gender" in nb.columns:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=nb, x="Gender")

    for bar in ax.patches:
        count = bar.get_height()

        if count:
            percentage = count * 100 / len(nb)
            ax.annotate(
                f"{percentage:.1f}%",
                (bar.get_x() + bar.get_width() / 2, count),
                ha="center",
                va="bottom"
            )

    plt.title("Gender Distribution")
    save_plot("gender_distribution.png")


# Average salary by specialization
if "Specialization" in nb.columns:
    avg_salary = (
        nb.groupby("Specialization")["Salary"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(14, 6))
    sns.barplot(
        x=avg_salary.index.astype(str),
        y=avg_salary.values
    )
    plt.xticks(rotation=90)
    plt.ylabel("Average Salary")
    plt.title("Average Salary by Specialization")
    save_plot("average_salary_by_specialization.png")


# Salary by categorical variables
categorical_plot_cols = [
    col for col in ["Gender", "Degree", "Specialization"]
    if col in nb.columns
]

if categorical_plot_cols:
    fig, axes = plt.subplots(
        1,
        len(categorical_plot_cols),
        figsize=(6 * len(categorical_plot_cols), 6),
        squeeze=False
    )

    for col, ax in zip(categorical_plot_cols, axes.flatten()):
        sns.boxplot(data=nb, x=col, y="Salary", ax=ax)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(f"Salary by {col}")

    save_plot("salary_boxplots.png")


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

X = nb.drop(columns=["Salary"])
y = nb["Salary"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=101
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Determine feature types from X.
categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ============================================================
# 6. DEFINE REGRESSION MODELS
# ============================================================

models = {
    "SGD Regressor": SGDRegressor(
        max_iter=5000,
        tol=1e-3,
        random_state=7
    ),

    "Decision Tree": DecisionTreeRegressor(
        min_samples_leaf=5,
        random_state=101
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=123,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=123,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=100
    ),

    "AdaBoost": AdaBoostRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=100
    )
}

# XGBoost is optional.
try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=100,
        n_jobs=-1
    )

except ImportError:
    print("\nXGBoost not installed; skipping it.")


# CatBoost is optional.
try:
    from catboost import CatBoostRegressor

    models["CatBoost"] = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=100,
        allow_writing_files=False
    )

except ImportError:
    print("\nCatBoost not installed; skipping it.")


# ============================================================
# 7. CROSS-VALIDATION MODEL COMPARISON
# ============================================================

scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

results = []
candidate_pipelines = {}

for model_name, model in models.items():

    print(f"\nEvaluating {model_name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", clone(model))
    ])

    try:
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        mean_rmse = -cv_results["test_rmse"].mean()
        std_rmse = cv_results["test_rmse"].std()
        mean_mae = -cv_results["test_mae"].mean()
        mean_r2 = cv_results["test_r2"].mean()

        results.append({
            "Model": model_name,
            "CV RMSE": mean_rmse,
            "CV RMSE Std": std_rmse,
            "CV MAE": mean_mae,
            "CV R2": mean_r2
        })

        candidate_pipelines[model_name] = pipeline

        print(f"CV RMSE: {mean_rmse:,.2f}")
        print(f"CV RMSE Std: {std_rmse:,.2f}")
        print(f"CV MAE:  {mean_mae:,.2f}")
        print(f"CV R2:   {mean_r2:.4f}")

    except Exception as error:
        print(f"Could not evaluate {model_name}: {error}")


if not results:
    raise RuntimeError(
        "No models completed successfully. Check the errors above."
    )


# ============================================================
# 8. UPGRADE 2A: LOG-TRANSFORMED SALARY TARGET
# ============================================================

# The log model wraps the entire preprocessing + regression pipeline.
# Predictions are automatically converted back to the original salary scale.

baseline_gb = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("model", clone(models["Gradient Boosting"]))
])

log_gb = TransformedTargetRegressor(
    regressor=clone(baseline_gb),
    func=np.log1p,
    inverse_func=np.expm1
)

print("\n--- Upgrade 2A: Gradient Boosting comparison ---")

for model_name, candidate in [
    ("Original Gradient Boosting", baseline_gb),
    ("Log-Transformed Gradient Boosting", log_gb)
]:

    cv_results = cross_validate(
        candidate,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    mean_rmse = -cv_results["test_rmse"].mean()
    std_rmse = cv_results["test_rmse"].std()
    mean_mae = -cv_results["test_mae"].mean()
    mean_r2 = cv_results["test_r2"].mean()

    results.append({
        "Model": model_name,
        "CV RMSE": mean_rmse,
        "CV RMSE Std": std_rmse,
        "CV MAE": mean_mae,
        "CV R2": mean_r2
    })

    candidate_pipelines[model_name] = clone(candidate)

    print(f"\n{model_name}")
    print(f"CV RMSE: {mean_rmse:,.2f}")
    print(f"CV RMSE Std: {std_rmse:,.2f}")
    print(f"CV MAE:  {mean_mae:,.2f}")
    print(f"CV R2:   {mean_r2:.4f}")


# ============================================================
# 9. SELECT THE BEST MODEL USING TRAINING CV ONLY
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV RMSE",
    ascending=True
).reset_index(drop=True)

print("\nComplete model comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    BASE_DIR / "model_comparison.csv",
    index=False
)

best_model_name = results_df.loc[0, "Model"]
best_model = candidate_pipelines[best_model_name]

print(f"\nSelected model based on CV RMSE: {best_model_name}")


# ============================================================
# 10. FINAL MODEL TRAINING AND TEST EVALUATION
# ============================================================

best_model.fit(X_train, y_train)

best_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, best_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
test_r2 = r2_score(y_test, best_predictions)

print("\nFinal held-out test performance:")
print(f"MAE:  {test_mae:,.2f}")
print(f"RMSE: {test_rmse:,.2f}")
print(f"R2:   {test_r2:.4f}")


# ============================================================
# 11. SAVE PREDICTIONS, METRICS, AND MODEL
# ============================================================

prediction_results = pd.DataFrame({
    "Actual Salary": y_test.to_numpy(),
    "Predicted Salary": best_predictions
})

prediction_results["Absolute Error"] = (
    prediction_results["Actual Salary"]
    - prediction_results["Predicted Salary"]
).abs()

print("\nFirst 10 predictions:")
print(prediction_results.head(10).round(2))

prediction_results.to_csv(
    BASE_DIR / "salary_predictions.csv",
    index=False
)

evaluation_report = pd.DataFrame([{
    "Selected Model": best_model_name,
    "Test MAE": test_mae,
    "Test RMSE": test_rmse,
    "Test R2": test_r2
}])

evaluation_report.to_csv(
    BASE_DIR / "final_test_metrics.csv",
    index=False
)

joblib.dump(
    best_model,
    BASE_DIR / "best_salary_prediction_model.joblib"
)

print("\nProject run completed successfully.")
print(f"Plots saved in: {PLOT_DIR}")
print("Model comparison saved to model_comparison.csv")
print("Predictions saved to salary_predictions.csv")
print("Test metrics saved to final_test_metrics.csv")
print("Final model saved to best_salary_prediction_model.joblib")
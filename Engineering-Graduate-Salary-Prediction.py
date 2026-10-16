from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

import joblib

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

# ============================================================
# 1. LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Engineering_graduate_salary.csv"

nb = pd.read_csv(DATA_PATH)

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
# 2. CHECK MISSING DATA AND UNIQUE VALUES
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


# ============================================================
# 3. REMOVE UNNECESSARY COLUMNS
# ============================================================

columns_to_drop = [
    "ID", "10board", "12graduation", "12board",
    "CollegeID", "CollegeCityID", "CollegeState",
    "CollegeCityTier", "DOB", "GraduationYear",
    "CollegeTier"
]

nb.drop(columns=columns_to_drop, errors="ignore", inplace=True)

if "Salary" not in nb.columns:
    raise ValueError("The dataset must contain a Salary column.")

# Standardize similar specialization names
if "Specialization" in nb.columns:
    nb["Specialization"] = nb["Specialization"].astype("string").str.strip()

    nb["Specialization"] = nb["Specialization"].str.replace(
        "electronics & instrumentation eng",
        "electronics and instrumentation engineering",
        case=False,
        regex=False
    )

    # Combine categories with 10 or fewer records
    counts = nb["Specialization"].value_counts()
    rare_categories = counts[counts <= 10].index

    nb.loc[
        nb["Specialization"].isin(rare_categories),
        "Specialization"
    ] = "other"

print("\nDataset after removing unnecessary columns:")
print(nb.head())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# Missing values heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(nb.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values")
save_plot("missing_values.png")


# 10th and 12th percentage scatter plot
if {"10percentage", "12percentage"}.issubset(nb.columns):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=nb,
        x="10percentage",
        y="12percentage"
    )
    plt.title("10th Percentage vs 12th Percentage")
    save_plot("percentage_scatter.png")


# Personality pair plot
personality_cols = [
    col for col in [
        "conscientiousness",
        "agreeableness",
        "extraversion",
        "nueroticism",
        "openess_to_experience",
        "Salary"
    ] if col in nb.columns
]

if len(personality_cols) >= 2:
    pair_plot = sns.pairplot(
        nb[personality_cols].dropna(),
        corner=True
    )
    pair_plot.fig.suptitle(
        "Personality Features and Salary",
        y=1.02
    )
    pair_plot.savefig(
        PLOT_DIR / "personality_pairplot.png",
        dpi=120
    )
    plt.show()
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


# Histograms
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
sns.histplot(nb["Salary"].dropna(), kde=True)
plt.title("Salary Distribution")

plt.subplot(1, 2, 2)
sorted_salary = np.sort(nb["Salary"].dropna().values)
plt.scatter(range(len(sorted_salary)), sorted_salary, s=10)
plt.title("Sorted Salary Curve")
plt.xlabel("Sorted Record")
plt.ylabel("Salary")

save_plot("salary_distribution.png")


# Gender count plot
if "Gender" in nb.columns:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=nb, x="Gender")

    for bar in ax.patches:
        count = bar.get_height()
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
categorical_plots = [
    col for col in ["Gender", "Degree", "Specialization"]
    if col in nb.columns
]

if categorical_plots:
    fig, axes = plt.subplots(
        1,
        len(categorical_plots),
        figsize=(6 * len(categorical_plots), 6),
        squeeze=False
    )

    for col, ax in zip(categorical_plots, axes.flatten()):
        sns.boxplot(data=nb, x=col, y="Salary", ax=ax)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(f"Salary by {col}")

    save_plot("salary_boxplots.png")


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

nb["Salary"] = pd.to_numeric(nb["Salary"], errors="coerce")
nb.dropna(subset=["Salary"], inplace=True)

X = nb.drop(columns=["Salary"])
y = nb["Salary"]

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

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=101
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 6. DEFINE REGRESSION MODELS
# ============================================================

models = {
    "SGD Regressor": SGDRegressor(
        max_iter=5000,
        tol=1e-3,
        eta0=0.0001,
        random_state=7
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=101
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=123,
        n_jobs=-1
    ),

    "Tuned Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_features=1.0,
        min_samples_leaf=2,
        random_state=123,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=100
    ),

    "AdaBoost": AdaBoostRegressor(
        random_state=100
    )
}

# XGBoost is optional
try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        objective="reg:squarederror",
        random_state=100
    )

except ImportError:
    print("\nXGBoost is not installed. Skipping XGBoost.")


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}
all_predictions = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, Y_train)

    predictions = pipeline.predict(X_test)

    # Root Mean Squared Error
    rmse = np.sqrt(
        mean_squared_error(Y_test, predictions)
    )

    # Mean Absolute Percentage Error
    nonzero_mask = Y_test.to_numpy() != 0

    if nonzero_mask.any():
        mape = mean_absolute_percentage_error(
            Y_test.to_numpy()[nonzero_mask],
            predictions[nonzero_mask]
        ) * 100
    else:
        mape = np.nan

    results.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAPE (%)": mape
    })

    trained_models[model_name] = pipeline
    all_predictions[model_name] = predictions

    print(f"RMSE: {rmse:,.2f}")

    if np.isfinite(mape):
        print(f"MAPE: {mape:.2f}%")
    else:
        print("MAPE: Not available")


# ============================================================
# 8. COMPARE MODEL PERFORMANCE
# ============================================================

results_df = pd.DataFrame(results)

results_df.sort_values(
    by="RMSE",
    ascending=True,
    inplace=True
)

results_df.reset_index(drop=True, inplace=True)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    BASE_DIR / "model_comparison.csv",
    index=False
)


# ============================================================
# 9. BEST MODEL AND PREDICTIONS
# ============================================================

best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]
best_predictions = all_predictions[best_model_name]

print("\nBest model based on test RMSE:", best_model_name)

prediction_results = pd.DataFrame({
    "Actual Salary": Y_test.to_numpy(),
    "Predicted Salary": best_predictions
})

print("\nFirst 10 predictions:")
print(prediction_results.head(10).round(2))

# Save predictions
prediction_results.to_csv(
    BASE_DIR / "salary_predictions.csv",
    index=False
)

# Save trained model and preprocessing pipeline
joblib.dump(
    best_model,
    BASE_DIR / "best_salary_prediction_model.joblib"
)

print("\nModel saved successfully.")
print("Results saved in:", BASE_DIR)
print("EDA plots saved in:", PLOT_DIR)
print("\nProject completed successfully.")
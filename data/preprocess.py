from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "educational-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


def _resolve_csv_path(csv_path: str) -> Path:
    candidate = Path(csv_path)
    if candidate.exists():
        return candidate

    here = Path(__file__).resolve()
    project_root = here.parent.parent

    alt = project_root / csv_path
    if alt.exists():
        return alt

    fallback = here.parent / "adult.csv"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find adult dataset at '{csv_path}', '{alt}', or '{fallback}'."
    )


def load_raw_adult(csv_path: str = "data/adult.csv") -> pd.DataFrame:
    """Load Adult dataset with explicit columns and skipped fake header row."""
    csv_file = _resolve_csv_path(csv_path)
    return pd.read_csv(csv_file, skiprows=1, names=ADULT_COLUMNS)


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print teammate-style quick checks before and after preprocessing."""
    print("Shape:", df.shape)
    print("Income brut:", df["income"].unique())
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])


def run_eda_plots(df: pd.DataFrame, output_dir: str | Path = "results") -> None:
    """
    Generate the same exploratory plots used by teammate code.
    Plots are saved in output_dir to keep script non-interactive friendly.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adult Income Dataset - Exploration", fontsize=16, fontweight="bold")

    df["income"].value_counts().plot(
        kind="bar", ax=axes[0, 0], color=["steelblue", "tomato"], edgecolor="black"
    )
    axes[0, 0].set_title("Distribution Income (cible)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=0)

    df["age"].hist(bins=30, ax=axes[0, 1], color="steelblue", edgecolor="black")
    axes[0, 1].set_title("Distribution de l'age")
    axes[0, 1].set_xlabel("Age")

    edu_income = df.groupby(["education", "income"]).size().unstack(fill_value=0)
    edu_income.plot(kind="bar", ax=axes[1, 0], colormap="coolwarm", edgecolor="black")
    axes[1, 0].set_title("Income par niveau d'education")
    axes[1, 0].tick_params(axis="x", rotation=45)

    gen_income = df.groupby(["gender", "income"]).size().unstack(fill_value=0)
    gen_income.plot(kind="bar", ax=axes[1, 1], colormap="Set2", edgecolor="black")
    axes[1, 1].set_title("Income par genre")
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(output_path / "eda_overview.png", dpi=150)
    plt.close(fig)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr().to_numpy()
    im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    ax_corr.set_title("Correlation entre variables numeriques")
    ax_corr.set_xticks(range(len(numeric_cols)))
    ax_corr.set_yticks(range(len(numeric_cols)))
    ax_corr.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax_corr.set_yticklabels(numeric_cols)
    fig_corr.tight_layout()
    fig_corr.savefig(output_path / "eda_correlation.png", dpi=150)
    plt.close(fig_corr)


def preprocess_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply full preprocessing pipeline before train/test split."""
    df = df.copy()

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.replace("?", np.nan, inplace=True)

    for col in ["workclass", "occupation", "native-country"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop("education", axis=1)
    df["income"] = (df["income"] == ">50K").astype(int)

    if verbose:
        print("Income:", df["income"].value_counts().to_dict())

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    present_num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    df[present_num_cols] = scaler.fit_transform(df[present_num_cols])

    return df


def get_data(
    csv_path: str = "data/adult.csv",
    with_eda: bool = False,
    eda_output_dir: str | Path = "results",
    verbose: bool = False,
    return_train_prior: bool = False,
    return_validation: bool = False,
):
    """
    Returns X_train, X_test, y_train, y_test as contiguous float64 numpy arrays.
    Keeps compatibility with the parallel trainer while integrating teammate preprocessing.
    """
    raw_df = load_raw_adult(csv_path)
    if verbose:
        print_dataset_overview(raw_df)

    if with_eda:
        run_eda_plots(raw_df, output_dir=eda_output_dir)

    df = preprocess_dataframe(raw_df, verbose=verbose)

    X = df.drop("income", axis=1)
    y = df["income"]

    X_fit, X_test, y_fit, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_fit, y_fit, test_size=0.2, random_state=42, stratify=y_fit
    )
    train_positive_prior = float(y_train_raw.mean())

    if verbose:
        print("Train(raw):", X_train_raw.shape)
        print("Val      :", X_val.shape)
        print("Test :", X_test.shape)
        print("-" * 44)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

    if verbose:
        print("Train apres SMOTE:", pd.Series(y_train).value_counts().to_dict())
        print("X_train shape:", X_train.shape)

    X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float64), dtype=np.float64)
    X_val_np = np.ascontiguousarray(X_val.to_numpy(dtype=np.float64), dtype=np.float64)
    X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float64), dtype=np.float64)
    y_train_np = np.ascontiguousarray(np.asarray(y_train, dtype=np.float64), dtype=np.float64)
    y_val_np = np.ascontiguousarray(np.asarray(y_val, dtype=np.float64), dtype=np.float64)
    y_test_np = np.ascontiguousarray(np.asarray(y_test, dtype=np.float64), dtype=np.float64)
    if return_validation and return_train_prior:
        return (
            X_train_np,
            X_val_np,
            X_test_np,
            y_train_np,
            y_val_np,
            y_test_np,
            train_positive_prior,
        )
    if return_validation:
        return X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np
    if return_train_prior:
        return X_train_np, X_test_np, y_train_np, y_test_np, train_positive_prior
    return X_train_np, X_test_np, y_train_np, y_test_np


if __name__ == "__main__":
    # Standalone check to reuse this file like the teammate notebook/script.
    get_data(verbose=True, with_eda=True)

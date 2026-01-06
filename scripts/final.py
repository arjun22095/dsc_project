import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.random_projection import GaussianRandomProjection

warnings.filterwarnings("ignore")

def load_or_generate_data(filepath='hmda_2017_nv_all.csv'):
    try:
        print(f"Attempting to load dataset from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Dataset Loaded Successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found.")
        exit(1)

def engineer_features(df):
    start_cols = df.shape[1]
    print("Engineering Features (creating ratios, cleaning names)...")
    print("Now ")
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]

    if "loan_amount_000s" in df.columns and "applicant_income_000s" in df.columns:
        loan_amt = pd.to_numeric(df["loan_amount_000s"], errors="coerce")
        income = pd.to_numeric(df["applicant_income_000s"], errors="coerce")
        ratio = loan_amt / income
        ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["loan_to_income_ratio"] = ratio

    if "co_applicant_sex_name" in df.columns:
        df["has_co_applicant"] = np.where(
            df["co_applicant_sex_name"].fillna("").astype(str).str.contains("No co-applicant", case=False),
            "No", "Yes"
        )

    end_cols = df.shape[1]
    print(f"   > Feature Count Change: {start_cols} -> {end_cols} (Added 'loan_to_income_ratio' & 'has_co_applicant')")

    return df


def remove_outliers(df, columns, factor=1.5):
    initial_rows = len(df)
    print(f"\n[Step: Outlier Removal] checking {columns}...")
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - factor*IQR)) & (df[col] <= (Q3 + factor*IQR))]
    
    final_rows = len(df)
    dropped = initial_rows - final_rows
    print(f"   > Removed {dropped} extreme rows based on IQR rules.")
    print(f"   > Rows remaining: {final_rows}")
    return df


def preprocess_data(df):
    print("\n--- PREPROCESSING START ---")
    print(f"Initial Shape: {df.shape}")
    
    df = engineer_features(df)

    df = remove_outliers(df, ["loan_amount_000s", "applicant_income_000s", "loan_to_income_ratio"])
    
    rows_before_na = len(df)
    df = df.dropna(thresh=len(df.columns) * 0.5)
    print(f"\n[Step: Drop NA] Removed {rows_before_na - len(df)} rows with excessive missing data.")

    def binarize(val):
        s = str(val)
        return 1 if ("Loan originated" in s or s == "1") else 0

    target_col = "action_taken_name" if "action_taken_name" in df.columns else "action_taken"
    y = df[target_col].apply(binarize).astype(int)
    class_counts = y.value_counts().sort_index()
    print("\n[Step: Class Distribution]")
    print(f"   > Class 0: {class_counts.get(0, 0)}")
    print(f"   > Class 1: {class_counts.get(1, 0)}")
    total_classes = class_counts.sum()
    if total_classes > 0:
        print(f"   > Balance: {class_counts.get(1,0)/total_classes:.4f} positive rate")


    leakage_keys = ['action_taken', 'denial_reason', 'purchaser_type', 'rate_spread',
                    'hoepa_status', 'lien_status', 'edit_status', 'sequence_number',
                    'application_date_indicator']
    
    cols_before_drop = df.shape[1]
    drop_cols = [col for col in df.columns if any(k in col for k in leakage_keys)]
    
    print(f"\n[Step: Leakage Removal]")
    print(f"   > Identified {len(drop_cols)} columns that cheat (contain target info).")
    print(f"   > Dropping: {drop_cols}")
    X = df.drop(columns=drop_cols)
    print(f"   > Columns reduced from {cols_before_drop} -> {X.shape[1]}")

    X_raw_train, X_raw_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n[Step: Train/Test Split]")
    print(f"   > Training Set: {X_raw_train.shape[0]} rows (The 80% used for JL/Coreset)")
    print(f"   > Test Set:     {X_raw_test.shape[0]} rows (The 20% held out)")

    numeric = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\n[Step: One-Hot Encoding]")
    print(f"   > Found {len(numeric)} numeric columns (kept as is).")
    print(f"   > Found {len(categorical)} categorical columns (will be exploded).")
    print(f"   > Note: A column with 50 categories becomes 50 new binary columns.")

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numeric),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical)
    ])

    X_train = preprocessor.fit_transform(X_raw_train)
    X_test  = preprocessor.transform(X_raw_test)

    if hasattr(X_train, "toarray"):
        X_train, X_test = X_train.toarray(), X_test.toarray()
    
    print(f"   > RESULT: Columns exploded from {len(numeric)+len(categorical)} -> {X_train.shape[1]}")
    print(f"   > This High Dimensionality ({X_train.shape[1]}) is why we use JL.")

    return X_train, X_test, y_train, y_test, X_raw_test


def compute_loss(model, X, y):
    probs = model.predict_proba(X)
    return log_loss(y, probs)


def train_evaluate(model, X_train, y_train, X_test, y_test, name="Model",
                   sample_weight=None, X_full_train=None, y_full_train=None):

    start = time.time()
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start

    if hasattr(model, "n_iter_"):
        iters = model.n_iter_[0] if hasattr(model.n_iter_, "__len__") else model.n_iter_
    else:
        iters = None

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    if X_full_train is not None:
        y_pred_train = model.predict(X_full_train)
        train_acc = accuracy_score(y_full_train, y_pred_train)
        train_f1 = f1_score(y_full_train, y_pred_train)
    else:
        y_pred_train = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train)

    print(f"[{name}] Time={train_time:.3f}s | Iters={iters} | TestAcc={test_acc:.4f} | TrainAcc={train_acc:.4f} | TestF1={test_f1:.4f} | TrainF1={train_f1:.4f}")

    return {
        "name": name,
        "model": model,
        "time": train_time,
        "iterations": iters,
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "test_f1": test_f1,
        "train_f1": train_f1
    }

def run_jl_experiment(X_train, X_test, y_train, y_test, full_loss):
    results = []
    n_features = X_train.shape[1]

    print("\n--- JL FEATURE SCALING EXPERIMENT ---")

    jl_k_values = [50, 100, 150, 200, 250]

    for k in jl_k_values:
        if k >= n_features:
            continue
        
        print(f"\n[Proof of Reduction] Reducing Features (Columns):")
        print(f"   Original dimensions: {X_train.shape[0]} rows x {n_features} columns")
        print(f"   Projected dimensions: {X_train.shape[0]} rows x {k} columns")
        print(f"   > The optimizer now handles {n_features/k:.1f}x fewer weights.")

        proj = GaussianRandomProjection(n_components=k, random_state=42)
        start = time.time()
        X_train_jl = proj.fit_transform(X_train)
        X_test_jl = proj.transform(X_test)
        proj_time = time.time() - start

        clf = LogisticRegression(max_iter=1000)
        res = train_evaluate(clf, X_train_jl, y_train, X_test_jl, y_test,
                             name=f"JL(k={k})")

        jl_loss = compute_loss(res['model'], X_train_jl, y_train)
        loss_ratio = jl_loss / full_loss

        res.update({
            "k": k,
            "train_loss": jl_loss,
            "loss_ratio": loss_ratio,
            "time": res["time"] + proj_time
        })

        results.append(res)

    return results


def build_coreset_and_train(X, y, X_test, y_test, full_loss):
    results = []
    
    m_sizes = [2000, 4000, 6000, 8000, 10000]

    print("\n--- CORESET EXPERIMENT ---")

    row_norms = np.linalg.norm(X, axis=1)**2
    probs = np.maximum(row_norms / row_norms.sum(), 1e-12)

    for m in m_sizes:
        core_start = time.time()
        if m > X.shape[0]:
            print(f"Warning: Coreset size m={m} is larger than training set size {X.shape[0]}. Skipping.")
            continue
            
        idx = np.random.choice(X.shape[0], size=m, replace=True, p=probs)

        X_core = X[idx]

        print(f"\n[Proof of Reduction] Reducing Dataset Size (Rows):")
        print(f"   Original dimensions: {X.shape[0]} rows x {X.shape[1]} columns")
        print(f"   Coreset dimensions:  {m} rows x {X.shape[1]} columns")
        print(f"   > The optimizer has {X.shape[0]/m:.1f}x fewer examples to process.")

        y_core = y.iloc[idx]
        weights = 1.0 / (m * probs[idx])
        core_time = time.time() - core_start

        clf = LogisticRegression(max_iter=1000)
        res = train_evaluate(clf, X_core, y_core, X_test, y_test,
                             name=f"Coreset(m={m})",
                             sample_weight=weights,
                             X_full_train=X, y_full_train=y)

        coreset_loss = compute_loss(res["model"], X, y)
        loss_ratio = coreset_loss / full_loss

        res.update({
            "m": m,
            "train_loss": coreset_loss,
            "loss_ratio": loss_ratio,
            "time": res["time"] + core_time
        })

        results.append(res)

    return results


def plot_loss_ratios(jl_results, coreset_results):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    ks = [r["k"] for r in jl_results]
    jl_ratios = [r["loss_ratio"] for r in jl_results]
    plt.plot(ks, jl_ratios, marker='o')
    plt.title("JL: Loss Ratio vs Dimension k")
    plt.xlabel("k (Projected Dimension)")
    plt.ylabel("Loss Ratio (L_JL / L_full)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    ms = [r["m"] for r in coreset_results]
    core_ratios = [r["loss_ratio"] for r in coreset_results]
    plt.plot(ms, core_ratios, marker='o', color='orange')
    plt.title("Coreset: Loss Ratio vs Size m")
    plt.xlabel("m (Coreset Size)")
    plt.ylabel("Loss Ratio (L_core / L_full)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_performance_metrics(full_res, jl_results, coreset_results):
    jl_ks = [r["k"] for r in jl_results]
    jl_acc_test = [r["test_accuracy"] for r in jl_results]
    jl_acc_train = [r["train_accuracy"] for r in jl_results]
    jl_f1_test = [r["test_f1"] for r in jl_results]
    jl_f1_train = [r["train_f1"] for r in jl_results]

    full_acc_test = full_res["test_accuracy"]
    full_acc_train = full_res["train_accuracy"]
    full_f1_test = full_res["test_f1"]
    full_f1_train = full_res["train_f1"]
    
    # Create 4 JL Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("JL Performance Comparison with Baseline (Full Model)", fontsize=16)

    axes[0, 0].plot(jl_ks, jl_acc_test, marker='o', label="JL Test Accuracy")
    axes[0, 0].axhline(full_acc_test, color='r', linestyle='--', label="Full Model Test Accuracy")
    axes[0, 0].set_title("JL: Test Accuracy vs k")
    axes[0, 0].set_xlabel("k (Projected Dimension)")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(jl_ks, jl_acc_train, marker='o', color='green', label="JL Train Accuracy")
    axes[0, 1].axhline(full_acc_train, color='r', linestyle='--', label="Full Model Train Accuracy")
    axes[0, 1].set_title("JL: Train Accuracy vs k")
    axes[0, 1].set_xlabel("k (Projected Dimension)")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # JL Test F1 Score
    axes[1, 0].plot(jl_ks, jl_f1_test, marker='o', color='purple', label="JL Test F1 Score")
    axes[1, 0].axhline(full_f1_test, color='r', linestyle='--', label="Full Model Test F1 Score")
    axes[1, 0].set_title("JL: Test F1 Score vs k")
    axes[1, 0].set_xlabel("k (Projected Dimension)")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # JL Train F1 Score
    axes[1, 1].plot(jl_ks, jl_f1_train, marker='o', color='orange', label="JL Train F1 Score")
    axes[1, 1].axhline(full_f1_train, color='r', linestyle='--', label="Full Model Train F1 Score")
    axes[1, 1].set_title("JL: Train F1 Score vs k")
    axes[1, 1].set_xlabel("k (Projected Dimension)")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    core_ms = [r["m"] for r in coreset_results]
    core_acc_test = [r["test_accuracy"] for r in coreset_results]
    core_acc_train = [r["train_accuracy"] for r in coreset_results]
    core_f1_test = [r["test_f1"] for r in coreset_results]
    core_f1_train = [r["train_f1"] for r in coreset_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Coreset Performance Comparison with Baseline (Full Model)", fontsize=16)

    # Coreset Test Accuracy
    axes[0, 0].plot(core_ms, core_acc_test, marker='o', label="Coreset Test Accuracy")
    axes[0, 0].axhline(full_acc_test, color='r', linestyle='--', label="Full Model Test Accuracy")
    axes[0, 0].set_title("Coreset: Test Accuracy vs m")
    axes[0, 0].set_xlabel("m (Coreset Size)")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Coreset Train Accuracy
    axes[0, 1].plot(core_ms, core_acc_train, marker='o', color='green', label="Coreset Train Accuracy")
    axes[0, 1].axhline(full_acc_train, color='r', linestyle='--', label="Full Model Train Accuracy")
    axes[0, 1].set_title("Coreset: Full Train Accuracy vs m")
    axes[0, 1].set_xlabel("m (Coreset Size)")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Coreset Test F1 Score
    axes[1, 0].plot(core_ms, core_f1_test, marker='o', color='purple', label="Coreset Test F1 Score")
    axes[1, 0].axhline(full_f1_test, color='r', linestyle='--', label="Full Model Test F1 Score")
    axes[1, 0].set_title("Coreset: Test F1 Score vs m")
    axes[1, 0].set_xlabel("m (Coreset Size)")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Coreset Train F1 Score
    axes[1, 1].plot(core_ms, core_f1_train, marker='o', color='orange', label="Coreset Train F1 Score")
    axes[1, 1].axhline(full_f1_train, color='r', linestyle='--', label="Full Model Train F1 Score")
    axes[1, 1].set_title("Coreset: Full Train F1 Score vs m")
    axes[1, 1].set_xlabel("m (Coreset Size)")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    df = load_or_generate_data()
    X_train, X_test, y_train, y_test, X_raw_test = preprocess_data(df)

    full_res = train_evaluate(LogisticRegression(max_iter=1000),
                              X_train, y_train, X_test, y_test,
                              name="Full")

    full_loss = compute_loss(full_res["model"], X_train, y_train)

    jl_results = run_jl_experiment(X_train, X_test, y_train, y_test, full_loss)

    coreset_results = build_coreset_and_train(
        X_train, y_train, X_test, y_test,
        full_loss
    )

    plot_loss_ratios(jl_results, coreset_results)
    
    plot_performance_metrics(full_res, jl_results, coreset_results)

    results = [full_res] + jl_results + coreset_results
    df_summary = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])
    print("\nFINAL SUMMARY:")
    print(df_summary.sort_values(by="time"))

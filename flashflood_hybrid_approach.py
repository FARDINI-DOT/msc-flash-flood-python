"""
Flashflood Classification Model (Hybrid Q-Percentile Approach)
==============================================================
Improvements for imbalanced data:
1. Hybrid approach: Train on Q-percentile labels, test on WL danger labels
2. Stratified train/validation/test split (no threshold tuning on test)
3. Threshold tuning on validation set only
4. Class-imbalance handling for all classifiers
5. Reporting with PR-AUC and test-only metrics
"""

import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# ===================== INTERACTIVE FEATURE SELECTION =====================
def ask_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True, "no": False, "n": False}
    prompt = " [Y/n]: " if default == "yes" else " [y/N]: "
    while True:
        choice = input(question + prompt).strip().lower()
        if choice == '' and default:
            return default == "yes"
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")

def interactive_feature_selection():
    """
    Interactive feature selection for the model.
    Returns selected features and a description string.
    """
    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)
    
    features = ["Q_direct_simulated"]
    feature_desc = []
    
    print("\nRecommended features are pre-selected. Change as needed:")
    
    # Q_direct_simulated is mandatory
    print("  Q_direct_simulated: ALWAYS INCLUDED (mandatory)")
    
    # dQ_dt (recommended)
    if ask_yes_no("\nInclude dQ_dt (discharge trend)?", default="yes"):
        features.append("dQ_dt")
        feature_desc.append("dqdt")
    
    # WL lags (recommended)
    if ask_yes_no("Include water level lag features (1d, 2d, 3d)?", default="yes"):
        features += ["wl_avg_daily_lag_1d", "wl_avg_daily_lag_2d", "wl_avg_daily_lag_3d"]
        feature_desc.append("wl")
    
    # day_of_year (not recommended if training per season)
    if ask_yes_no("Include day_of_year? (not recommended if training per season)", default="no"):
        features.append("day_of_year")
        feature_desc.append("doy")
    
    # Precipitation lags (not recommended - redundant with Q)
    if ask_yes_no("Include precipitation lag features? (not recommended - redundant with Q)", default="no"):
        features += [
            "precip_lag_1d",
            "precip_lag_2d",
            "precip_lag_3d",
            "precip_lag_4d",
            "precip_lag_5d",
        ]
        feature_desc.append("precip")
    
    if len(features) == 1:
        feature_desc.append("only_q")
    
    desc = "_".join(feature_desc) if feature_desc else "only_q"
    
    print(f"\nSelected features ({len(features)}): {features}")
    print(f"Description: {desc}")
    
    return features, desc

print("=" * 70)
print("FLASHFLOOD CLASSIFICATION MODEL (HYBRID Q-PERCENTILE APPROACH)")
print("=" * 70)

ARTIFACT_ROOT = Path(r"D:\Masters\data\predicting_flashflood_events_codes\ML_training\tp25tb110")
PLOT_ROOT = ARTIFACT_ROOT / "CLASSIFICATION RESULTS"


# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
print("\nLoading data...")

data_file = r"D:\Masters\data\predicting_flashflood_events_codes\simulated_q_tp25tb110.xlsx"

try:
    df = pd.read_excel(data_file, parse_dates=["Date Time"])
    print(f"Loaded {len(df)} rows")
except Exception:
    print(f"Could not load {data_file}")
    print("Please update data_file path in this script.")
    raise SystemExit(1)

water_level_candidates = ["WL (mMSL)", "WL", "water_level", "Water Level"]
water_level_col = next((c for c in water_level_candidates if c in df.columns), None)
if water_level_col is None:
    print("Could not find water level column in data_file.")
    print(f"Looked for one of: {water_level_candidates}")
    raise SystemExit(1)
print(f"Using water level column from data file: {water_level_col}")


# ==============================================================================
# STEP 2: FEATURE ENGINEERING
# ==============================================================================
print("\nEngineering features...")

df["dQ_dt"] = df["Q_direct_simulated"].diff() / 1.0
df["day_of_year"] = df["Date Time"].dt.dayofyear

# Build daily-average observed water level from data_file and lag it by 1-3 days.
df_wl_daily = (
    df.assign(date=df["Date Time"].dt.floor("D"))
    .groupby("date", as_index=False)[water_level_col]
    .mean()
    .rename(columns={water_level_col: "wl_avg_daily"})
)
for lag_day in [1, 2, 3]:
    df_wl_daily[f"wl_avg_daily_lag_{lag_day}d"] = df_wl_daily["wl_avg_daily"].shift(lag_day)

df["date"] = df["Date Time"].dt.floor("D")
df = df.merge(
    df_wl_daily[["date", "wl_avg_daily_lag_1d", "wl_avg_daily_lag_2d", "wl_avg_daily_lag_3d"]],
    on="date",
    how="left",
)
df = df.drop(columns=["date"])

for lag in range(1, 6):
    df[f"precip_lag_{lag}d"] = df["corr_precip"].rolling(window=lag * 24, min_periods=1).sum()

df = df.dropna().copy()
print(f"Features created. {len(df)} rows after cleaning.")


# ==============================================================================
# STEP 3: CREATE HYBRID LABELS (Q-based for training, WL-based for testing)
# ==============================================================================
print("\nCreating hybrid labels...")

# Add season column for stratification by season
df['season'] = df['Date Time'].dt.month.apply(
    lambda m: 'MONSOON' if m in [6, 7] else 'PREMONSOON'
)

# TRAINING LABEL: Q percentile (balanced, easy to learn)
df['flashflood_train'] = 0
for season in ['PREMONSOON', 'MONSOON']:
    mask = df['season'] == season
    q_95 = df[mask]['Q_direct_simulated'].quantile(0.95)  # Use 95th percentile to get ~10% positives   
    df.loc[mask & (df['Q_direct_simulated'] > q_95), 'flashflood_train'] = 1

# TESTING LABEL: WL danger level (operational, real floods)
danger_levels = {'PREMONSOON': 10.69, 'MONSOON': 12.34}
df['flashflood_test'] = 0
for season, danger_wl in danger_levels.items():
    mask = df['season'] == season
    df.loc[mask & (df[water_level_col] > danger_wl), 'flashflood_test'] = 1

print(f"\nLabel distribution:")
print(f"  Training label (Q > 95th percentile):")
print(f"    Floods: {df['flashflood_train'].sum()} ({df['flashflood_train'].mean() * 100:.2f}%)")
print(f"    Non-floods: {len(df) - df['flashflood_train'].sum()} ({(1 - df['flashflood_train'].mean()) * 100:.2f}%)")
print(f"  Testing label (WL > danger level):")
print(f"    Floods: {df['flashflood_test'].sum()} ({df['flashflood_test'].mean() * 100:.2f}%)")
print(f"    Non-floods: {len(df) - df['flashflood_test'].sum()} ({(1 - df['flashflood_test'].mean()) * 100:.2f}%)")


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1-score."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []

    for thresh in thresholds:
        y_pred_thresh = (y_proba > thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx]), thresholds, f1_scores


def evaluate_with_threshold(y_true, y_proba, threshold):
    """Evaluate binary metrics at a selected threshold."""
    y_pred = (y_proba > threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "y_pred": y_pred,
    }


def run_period_model(df_period, period_label, output_tag, season_folder, feature_columns, experiment_name, per_model_plot_dir=None):
    """
    Train and evaluate models for a specific period (season).
    
    Uses hybrid approach:
    - Trains on Q-percentile labels (balanced, ~10% positive)
    - Tests on WL danger labels (operational, ~1.5% positive)
    """
    print("\n" + "=" * 70)
    print(f"RUNNING PERIOD: {period_label}")
    print("=" * 70)

    if df_period.empty:
        print(f"Skipping {period_label}: no rows found.")
        return

    X = df_period[feature_columns]
    
    # Use Q-percentile labels for training split (balanced)
    y_train_label = df_period["flashflood_train"]
    # Use WL danger labels for testing (operational)
    y_test_label = df_period["flashflood_test"]

    print("\nTraining label (Q-percentile) distribution:")
    print(y_train_label.value_counts())
    print(f"  Floods: {int(y_train_label.sum())} ({y_train_label.mean() * 100:.2f}%)")
    print(f"  Non-floods: {int(len(y_train_label) - y_train_label.sum())} ({(1 - y_train_label.mean()) * 100:.2f}%)")
    
    print("\nTesting label (WL danger) distribution:")
    print(y_test_label.value_counts())
    print(f"  Floods: {int(y_test_label.sum())} ({y_test_label.mean() * 100:.2f}%)")
    print(f"  Non-floods: {int(len(y_test_label) - y_test_label.sum())} ({(1 - y_test_label.mean()) * 100:.2f}%)")

    if y_train_label.nunique() < 2:
        print(f"Skipping {period_label}: training labels only have one class.")
        return

    class_counts = y_train_label.value_counts()
    if class_counts.min() < 3:
        print(f"Skipping {period_label}: at least one class has fewer than 3 samples.")
        return

    print("\nSplitting data: 60% train, 20% validation, 20% test (stratified on training label)...")

    try:
        X_trainval, X_test, y_trainval, _ = train_test_split(
            X,
            y_train_label,  # Split based on Q-percentile label
            test_size=0.20,
            random_state=42,
            stratify=y_train_label,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.25,
            random_state=42,
            stratify=y_trainval,
        )
    except ValueError as ex:
        print(f"Skipping {period_label}: split failed -> {ex}")
        return

    # Get the corresponding test labels for evaluation
    y_test_official = df_period.loc[X_test.index, 'flashflood_test']

    print(f"Training:   {len(X_train)} (Q-label)")
    print(f"Validation: {len(X_val)} (Q-label)")
    print(f"Testing:    {len(X_test)} (will evaluate on WL-label)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Balanced sample weights for models without class_weight support.
    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=3000,
            C=0.5,
        ),
    }

    print("\nTraining models on Q-percentile labels...")
    trained_models = {}
    val_probas = {}
    test_probas = {}

    for model_name, model in models.items():
        # Train on Q-percentile labels (balanced)
        model.fit(X_train_scaled, y_train)

        trained_models[model_name] = model
        val_probas[model_name] = model.predict_proba(X_val_scaled)[:, 1]
        test_probas[model_name] = model.predict_proba(X_test_scaled)[:, 1]
        print(f"  {model_name} trained")

        # ========== INDIVIDUAL MODEL PLOTS ========== #
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion matrix (using test official labels for evaluation)
        best_thresh, best_f1_val, all_thresh, all_f1 = find_optimal_threshold(y_val, val_probas[model_name])
        y_pred = (test_probas[model_name] > best_thresh).astype(int)
        cm = confusion_matrix(y_test_official, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0], xticklabels=["No Flood", "Flood"], yticklabels=["No Flood", "Flood"])
        axes[0, 0].set_title(f"Confusion Matrix ({model_name})\n(thr={best_thresh:.2f})", fontweight="bold")
        axes[0, 0].set_ylabel("True Label (WL danger)")
        axes[0, 0].set_xlabel("Predicted Label")
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test_official, test_probas[model_name])
        auc = roc_auc_score(y_test_official, test_probas[model_name])
        axes[0, 1].plot(fpr, tpr, label=f"AUC={auc:.3f}", linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].set_title("ROC Curve (Test)", fontweight="bold")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test_official, test_probas[model_name])
        axes[1, 0].plot(recall, precision, linewidth=2)
        axes[1, 0].set_xlabel("Recall (POD)")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Precision-Recall Curve (Test on WL danger)", fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 vs threshold
        axes[1, 1].plot(all_thresh, all_f1, linewidth=2)
        axes[1, 1].scatter([best_thresh], [best_f1_val], s=80, edgecolor="black", linewidth=1.5)
        axes[1, 1].set_xlabel("Threshold")
        axes[1, 1].set_ylabel("Validation F1 (on Q-label)")
        axes[1, 1].set_title("Threshold Optimization (Validation)", fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)

        # Feature importance or coefficients
        if model_name == "Random Forest":
            importances = model.feature_importances_
            print("\nFeature importances (Random Forest):")
            for fname, imp in zip(feature_columns, importances):
                print(f"  {fname}: {imp:.4f}")
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            pd.Series(importances, index=feature_columns).sort_values().plot(kind="barh", ax=ax2, color="steelblue")
            ax2.set_title("Feature Importances (Random Forest)")
            ax2.set_xlabel("Importance")
            plt.tight_layout()
            if per_model_plot_dir is not None:
                per_model_plot_dir.mkdir(parents=True, exist_ok=True)
                plot_file2 = per_model_plot_dir / f"{model_name.replace(' ', '_')}_feature_importance.png"
                plt.savefig(plot_file2, dpi=150, bbox_inches="tight")
                print(f"Saved: {plot_file2}")
            plt.close(fig2)
            
        elif model_name == "Logistic Regression":
            coefs = model.coef_[0]
            print("\nCoefficient values (Logistic Regression):")
            for fname, coef in zip(feature_columns, coefs):
                print(f"  {fname}: {coef:.4f}")
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            pd.Series(coefs, index=feature_columns).sort_values().plot(kind="barh", ax=ax2, color="orange")
            ax2.set_title("Coefficient Values (Logistic Regression)")
            ax2.set_xlabel("Coefficient")
            plt.tight_layout()
            if per_model_plot_dir is not None:
                per_model_plot_dir.mkdir(parents=True, exist_ok=True)
                plot_file2 = per_model_plot_dir / f"{model_name.replace(' ', '_')}_coefficients.png"
                plt.savefig(plot_file2, dpi=150, bbox_inches="tight")
                print(f"Saved: {plot_file2}")
            plt.close(fig2)

    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION ON VALIDATION SET (Q-label)")
    print("=" * 70)

    optimal_thresholds = {}
    test_results = {}

    for model_name in models.keys():
        y_val_proba = val_probas[model_name]
        y_test_proba = test_probas[model_name]

        best_thresh, best_f1_val, all_thresh, all_f1 = find_optimal_threshold(y_val, y_val_proba)

        val_f1_at_default = f1_score(y_val, (y_val_proba > 0.5).astype(int), zero_division=0)
        # Evaluate on test official (WL danger) labels
        test_eval = evaluate_with_threshold(y_test_official, y_test_proba, best_thresh)

        optimal_thresholds[model_name] = {
            "threshold": best_thresh,
            "val_best_f1": best_f1_val,
            "val_f1_default": float(val_f1_at_default),
            "all_thresholds": all_thresh,
            "all_f1_scores": all_f1,
        }
        test_results[model_name] = test_eval

        print(f"\n{model_name}:")
        print(f"  Validation F1 @ 0.50 (Q-label): {val_f1_at_default:.4f}")
        print(f"  Validation best threshold: {best_thresh:.2f}")
        print(f"  Validation best F1 (Q-label): {best_f1_val:.4f}")

    print("\n" + "=" * 70)
    print("TEST PERFORMANCE (EVALUATED ON WL DANGER LABELS)")
    print("=" * 70)

    for model_name in models.keys():
        r = test_results[model_name]
        t = optimal_thresholds[model_name]["threshold"]

        print(f"\n{model_name} (threshold = {t:.2f}):")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall (POD): {r['recall']:.4f}")
        print(f"  F1-Score:  {r['f1']:.4f}")
        print(f"  ROC-AUC:   {r['roc_auc']:.4f}")
        print(f"  PR-AUC:    {r['pr_auc']:.4f}")

        print("\n  Classification Report (test on WL danger):")
        print(
            classification_report(
                y_test_official,
                r["y_pred"],
                target_names=["No Flood", "Flood"],
                zero_division=0,
            )
        )

    # Choose the best model by validation F1.
    best_model_name = max(optimal_thresholds, key=lambda k: optimal_thresholds[k]["val_best_f1"])
    best_model = trained_models[best_model_name]
    best_threshold = optimal_thresholds[best_model_name]["threshold"]

    print("\n" + "=" * 70)
    print(f"BEST MODEL BY VALIDATION F1: {best_model_name}")
    print(f"Threshold: {best_threshold:.4f}")
    print("=" * 70)

    feature_importance = None
    if best_model_name == "Random Forest":
        feature_importance = pd.DataFrame(
            {"Feature": feature_columns, "Importance": best_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE (Random Forest)")
        print("=" * 70)
        print(feature_importance.to_string(index=False))

    print("\nCreating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1) Confusion matrix for best model on test.
    ax = axes[0, 0]
    best_y_pred = test_results[best_model_name]["y_pred"]
    cm = confusion_matrix(y_test_official, best_y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No Flood", "Flood"],
        yticklabels=["No Flood", "Flood"],
    )
    ax.set_title(f"Confusion Matrix ({best_model_name}, thr={best_threshold:.2f})", fontweight="bold")
    ax.set_ylabel("True Label (WL danger)")
    ax.set_xlabel("Predicted Label")

    # 2) ROC curves on test.
    ax = axes[0, 1]
    for model_name in models.keys():
        fpr, tpr, _ = roc_curve(y_test_official, test_probas[model_name])
        auc = roc_auc_score(y_test_official, test_probas[model_name])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) F1-vs-threshold on validation.
    ax = axes[0, 2]
    for model_name in models.keys():
        opt = optimal_thresholds[model_name]
        ax.plot(opt["all_thresholds"], opt["all_f1_scores"], label=model_name, linewidth=2)
        ax.scatter([opt["threshold"]], [opt["val_best_f1"]], s=80, edgecolor="black", linewidth=1.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Validation F1 (Q-label)")
    ax.set_title("Threshold Optimization (Validation)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) Feature importance
    ax = axes[1, 0]
    if feature_importance is not None:
        feature_importance.plot(x="Feature", y="Importance", kind="barh", ax=ax, color="steelblue", legend=False)
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")
    else:
        ax.text(0.5, 0.5, "Feature importance shown\nwhen best model is RF", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Feature Importance", fontweight="bold")

    # 5) PR curves on test.
    ax = axes[1, 1]
    for model_name in models.keys():
        precision, recall, _ = precision_recall_curve(y_test_official, test_probas[model_name])
        ax.plot(recall, precision, label=model_name, linewidth=2)
    ax.set_xlabel("Recall (POD)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Test on WL danger)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6) Summary bars on test.
    ax = axes[1, 2]
    model_names = list(models.keys())
    metrics_data = {
        "F1": [test_results[m]["f1"] for m in model_names],
        "PR-AUC": [test_results[m]["pr_auc"] for m in model_names],
        "Recall (POD)": [test_results[m]["recall"] for m in model_names],
    }

    x = np.arange(len(model_names))
    width = 0.25
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Test on WL danger)", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    plot_dir = PLOT_ROOT / experiment_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / f"flashflood_classification_results_hybrid_{output_tag}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_file}")

    print("\nSaving artifacts...")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    period_dir = ARTIFACT_ROOT / experiment_name / season_folder
    period_dir.mkdir(parents=True, exist_ok=True)

    all_models_summary = {
        "experiment": experiment_name,
        "period": period_label,
        "output_tag": output_tag,
        "season_folder": season_folder,
        "approach": "HYBRID_Q_PERCENTILE",
        "training_label": "Q > 90th percentile (balanced)",
        "testing_label": "WL > danger level (operational)",
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "feature_columns": feature_columns,
        "models": {},
    }

    for model_name, model in trained_models.items():
        model_dir = period_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / "model.pkl"
        scaler_file = model_dir / "scaler.pkl"
        info_file = model_dir / "model_info.json"

        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)

        model_info = {
            "period": period_label,
            "output_tag": output_tag,
            "season_folder": season_folder,
            "experiment": experiment_name,
            "model_name": model_name,
            "is_best_model": model_name == best_model_name,
            "approach": "HYBRID_Q_PERCENTILE",
            "training_on": "Q percentile labels (balanced)",
            "testing_on": "WL danger labels (operational)",
            "threshold_info": {
                "optimal_threshold": optimal_thresholds[model_name]["threshold"],
                "val_best_f1": optimal_thresholds[model_name]["val_best_f1"],
                "val_f1_default": optimal_thresholds[model_name]["val_f1_default"],
                "all_thresholds": optimal_thresholds[model_name]["all_thresholds"].tolist(),
                "all_f1_scores": optimal_thresholds[model_name]["all_f1_scores"],
            },
            "feature_columns": feature_columns,
            "test_metrics": {
                "accuracy": test_results[model_name]["accuracy"],
                "precision": test_results[model_name]["precision"],
                "recall": test_results[model_name]["recall"],
                "f1": test_results[model_name]["f1"],
                "roc_auc": test_results[model_name]["roc_auc"],
                "pr_auc": test_results[model_name]["pr_auc"],
            },
        }

        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2)

        all_models_summary["models"][model_name] = {
            "folder": str(model_dir),
            "optimal_threshold": optimal_thresholds[model_name]["threshold"],
            "test_f1": test_results[model_name]["f1"],
            "test_pr_auc": test_results[model_name]["pr_auc"],
            "test_recall_pod": test_results[model_name]["recall"],
        }

        print(f"Saved: {model_file}")
        print(f"Saved: {scaler_file}")
        print(f"Saved: {info_file}")

    summary_file = period_dir / "all_models_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_models_summary, f, indent=2)
    print(f"Saved: {summary_file}")


# ============================================================================== 
# STEP 3+: INTERACTIVE FEATURE SELECTION & EXPERIMENTS
# ==============================================================================
period_configs = [
    ("March-May (PREMONSOON)", [3, 4, 5], "march_may", "PREMONSOON"),
    ("June-July (MONSOON)", [6, 7], "june_july", "MONSOON"),
]

print("\n" + "=" * 70)
print("INTERACTIVE FEATURE SELECTION")
print("=" * 70)
user_features, user_desc = interactive_feature_selection()

experiment_configs = [
    (f"HYBRID_Q_USER_SELECTED_{user_desc.upper()}", user_features, user_desc),
    ("HYBRID_Q_SIMULATED_DISCHARGE_ONLY", ["Q_direct_simulated"], "sim_q_only"),
]

for experiment_name, feature_columns, output_suffix in experiment_configs:
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Features used ({len(feature_columns)}): {feature_columns}")
    print(f"Approach: HYBRID Q-PERCENTILE")
    print(f"  - Training label: Q > 90th percentile (balanced)")
    print(f"  - Testing label: WL > danger level (operational)")
    print("=" * 70)

    for period_label, months, output_tag, season_folder in period_configs:
        df_period = df[df["Date Time"].dt.month.isin(months)].copy()
        print(f"\nSelected {len(df_period)} rows for {period_label} (months={months})")
        
        per_model_plot_dir = PLOT_ROOT / experiment_name / season_folder / f"{output_tag}_{output_suffix}"
        run_period_model(
            df_period,
            period_label,
            f"{output_tag}_{output_suffix}",
            season_folder,
            feature_columns,
            experiment_name,
            per_model_plot_dir=per_model_plot_dir,
        )

print("\n" + "=" * 70)
print("COMPLETE: Hybrid Q-percentile runs finished for all periods")
print("=" * 70)
print("\nSummary:")
print("  - Models trained on Q-percentile labels (Q > 95th percentile)")
print("  - Models evaluated on WL danger labels (WL > danger level)")
print("  - This hybrid approach solves class imbalance in training")
print("  - While maintaining operational relevance in testing")

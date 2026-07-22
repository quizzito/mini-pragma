"""
M4: XGBoost baselines for the 3 downstream tasks (credit_default, fraud,
engagement), using hand-engineered features. This is the "task-specific
model" reference row we'll compare the foundation model against later,
same role as the grey bars in Figure 1 of the PRAGMA paper.
"""
import pandas as pd

events = pd.read_parquet("data_gen/output/events.parquet")
profiles = pd.read_parquet("data_gen/output/profiles.parquet")

print(f"Loaded {len(profiles)} profiles, {len(events)} events")


def compute_total_spend(user_events: pd.DataFrame) -> float:
    """Total amount spent on card_payments for one user."""
    card_payments = user_events[user_events["type"] == "card_payment"]
    return card_payments["amount"].sum()

def compute_features(events: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hand-engineered features for every user, using groupby (efficient,
    vectorized — not a Python loop per user). Returns one row per user.
    """
    # Total card spend per user
    card_payments = events[events["type"] == "card_payment"]
    total_spend = card_payments.groupby("user_id")["amount"].sum().rename("total_spend")

    # Total topup amount per user
    topups = events[events["type"] == "topup"]
    total_topup = topups.groupby("user_id")["amount"].sum().rename("total_topup")

    # Event counts by type (one column per type)
    event_counts = events.groupby(["user_id", "type"]).size().unstack(fill_value=0)
    event_counts.columns = [f"count_{col}" for col in event_counts.columns]

    # Total number of events per user
    total_events = events.groupby("user_id").size().rename("total_events")

    # Combine all features into one table, starting from profiles (so every
    # user is included even if some feature is missing for them)
    features = profiles[["user_id", "balance", "tenure_months"]].set_index("user_id")
    features = features.join(total_spend).join(total_topup).join(event_counts).join(total_events)

    # Users with no card_payments/topups will have NaN here (not zero) —
    # fill those with 0, since "no transactions" genuinely means zero spend
    features = features.fillna(0)

    return features

from sklearn.model_selection import train_test_split


def prepare_dataset(features: pd.DataFrame, profiles: pd.DataFrame, label_column: str):
    """
    Join features with one label column, then split into train/test sets.
    `stratify` ensures both splits have a similar proportion of positive
    labels — important since our labels are imbalanced (e.g. fraud is rare).
    """
    labels = profiles.set_index("user_id")[label_column]
    data = features.join(labels)

    X = data.drop(columns=[label_column])
    y = data[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def train_and_evaluate(X_train, X_test, y_train, y_test, task_name: str) -> dict:
    """
    Train an XGBoost classifier and evaluate with ROC-AUC and PR-AUC —
    the same two metrics used throughout the PRAGMA paper's Table 2.
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # predict_proba gives probability of class 1 (the positive/rare class)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print(f"\n--- {task_name} ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    return {"task": task_name, "roc_auc": roc_auc, "pr_auc": pr_auc, "model": model}

def save_results(results: list[dict], output_path: str = "results/baseline_results.csv") -> None:
    """Save baseline results to CSV — this becomes the reference row M6 compares against."""
    import os
    os.makedirs("results", exist_ok=True)

    rows = [{"task": r["task"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"]} for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved baseline results to {output_path}")
    print(df)

if __name__ == "__main__":
    features = compute_features(events, profiles)

    results = []
    for task in ["credit_default", "fraud", "engagement"]:
        X_train, X_test, y_train, y_test = prepare_dataset(features, profiles, task)
        result = train_and_evaluate(X_train, X_test, y_train, y_test, task)
        results.append(result)

    save_results(results)
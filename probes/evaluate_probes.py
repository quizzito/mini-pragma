"""
M6 core evaluation: train a logistic regression probe on top of the FROZEN
pretrained embeddings, per task, and compare directly against the M4
XGBoost baselines. This mirrors the paper's own embedding-probe protocol
(§3.1.1).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

embeddings = np.load("results/user_embeddings.npy")
embedding_user_ids = np.load("results/user_embeddings_user_ids.npy")
profiles = pd.read_parquet("data_gen/output/profiles.parquet")

# Build a lookup so embeddings and labels line up correctly by user_id,
# rather than assuming row order matches (never assume -- verify)
embeddings_df = pd.DataFrame(embeddings, index=embedding_user_ids)
embeddings_df = embeddings_df.loc[profiles["user_id"]]  # reorder to match profiles exactly

print(f"Embeddings shape (reordered to match profiles): {embeddings_df.shape}")
assert (embeddings_df.index.values == profiles["user_id"].values).all(), "User ID mismatch!"
print("Confirmed: embeddings correctly aligned with profile user_ids.\n")


def evaluate_probe(embeddings_df, profiles, label_column, task_name):
    X = embeddings_df.values
    y = profiles.set_index("user_id").loc[embeddings_df.index, label_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train_scaled, y_train)

    y_pred_proba = probe.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print(f"--- {task_name} (embedding probe) ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    np.save(f"results/probe_{task_name}_y_test.npy", y_test)
    np.save(f"results/probe_{task_name}_y_pred.npy", y_pred_proba)

    return {"task": task_name, "roc_auc": roc_auc, "pr_auc": pr_auc}


if __name__ == "__main__":
    results = []
    for task in ["credit_default", "fraud", "engagement"]:
        result = evaluate_probe(embeddings_df, profiles, task, task)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/probe_results.csv", index=False)
    print(f"\nSaved probe results to results/probe_results.csv")
    print(results_df)
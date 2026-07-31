"""
M6 sanity check: before running real downstream evaluation, confirm the
embeddings actually vary meaningfully across users -- not collapsed to
near-identical vectors (which would mean the model learned nothing useful),
and not pure noise unrelated to any real signal.
"""
import numpy as np

embeddings = np.load("results/user_embeddings.npy")
user_ids = np.load("results/user_embeddings_user_ids.npy")

print(f"Embeddings shape: {embeddings.shape}")

print("\n--- Basic statistics across all embeddings ---")
print(f"Mean (across all users, all dims): {embeddings.mean():.4f}")
print(f"Std (across all users, all dims): {embeddings.std():.4f}")
print(f"Min: {embeddings.min():.4f}, Max: {embeddings.max():.4f}")

print("\n--- Per-user variation check ---")
# If every user had an IDENTICAL embedding, the std across users (per dimension)
# would be ~0. We want to see real variation.
per_dimension_std = embeddings.std(axis=0)  # std across the 2000 users, for each of the 32 dims
print(f"Std across users, per dimension (first 8 dims): {per_dimension_std[:8].round(4)}")
print(f"Average per-dimension std: {per_dimension_std.mean():.4f}")

print("\n--- Similarity check: are two random users different? ---")
from numpy.linalg import norm

user_a = embeddings[0]
user_b = embeddings[1]
cosine_similarity = np.dot(user_a, user_b) / (norm(user_a) * norm(user_b))
print(f"Cosine similarity between user 0 and user 1: {cosine_similarity:.4f}")
print("(1.0 = identical, 0.0 = unrelated, -1.0 = opposite -- we want something clearly less than 1.0)")

print("\n--- Similarity across several pairs ---")
profiles = __import__("pandas").read_parquet("data_gen/output/profiles.parquet")

pairs_to_check = [(0, 1), (0, 500), (0, 1999), (100, 1500), (50, 51)]
for i, j in pairs_to_check:
    sim = np.dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]))
    profile_i = profiles.iloc[i]
    profile_j = profiles.iloc[j]
    print(f"User {i} vs User {j}: cosine_sim={sim:.4f} | "
          f"balances: {profile_i['balance']:.0f} vs {profile_j['balance']:.0f}")
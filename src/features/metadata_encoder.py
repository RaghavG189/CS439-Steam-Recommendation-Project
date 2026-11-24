import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def build_game_feature_matrix(
    raw_dir="data/raw",
    processed_dir="data/processed",
    use_metadata_json=True,
):
    out_path = Path(processed_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    games_clean_file = out_path / "games_clean.parquet"
    mappings_file = out_path / "id_mappings.json"

    # 1. Align strictly to Raghav's id_mappings.json if available; otherwise test with raw/games.csv
    if games_clean_file.exists() and mappings_file.exists():
        games = pd.read_parquet(games_clean_file)
        with open(mappings_file, "r") as f:
            mappings = json.load(f)
        app_to_idx = {int(k): v for k, v in mappings["app_to_idx"].items()}
        games = games[games["app_id"].isin(app_to_idx)].copy()
        games["app_idx"] = games["app_id"].map(app_to_idx)
        games = games.sort_values("app_idx").reset_index(drop=True)
        print(f"Aligned {len(games):,} games to Raghav's id_mappings.json.")
    else:
        games = pd.read_csv(f"{raw_dir}/games.csv")
        games["title"] = games["title"].str.strip()
        games["app_idx"] = np.arange(len(games))
        print(f"Loaded {len(games):,} games from raw/games.csv (temporary mode).")

    # 2. Build text corpus (title + rating, plus tags/description if games_metadata.json exists)
    text_series = (
        games["title"].fillna("")
        + " rating_"
        + games["rating"].astype(str).str.replace(" ", "_")
    )

    meta_json_path = Path(raw_dir) / "games_metadata.json"
    if use_metadata_json and meta_json_path.exists():
        meta = pd.read_json(meta_json_path, lines=True)[["app_id", "tags", "description"]]
        meta["tags_str"] = meta["tags"].apply(
            lambda t: " ".join(t) if isinstance(t, list) else ""
        )
        games = games.merge(meta[["app_id", "tags_str", "description"]], on="app_id", how="left")
        text_series = (
            text_series
            + " "
            + games["tags_str"].fillna("")
            + " "
            + games["description"].fillna("")
        )
        print("Merged tags & description from games_metadata.json.")

    # 3. TF-IDF encoding
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(text_series)

    # 4. Normalize numeric metadata & combine with OS/Steam Deck flags
    scaler = MinMaxScaler()
    num_cols = scaler.fit_transform(games[["positive_ratio", "price_final"]].fillna(0))
    bool_cols = games[["win", "mac", "linux", "steam_deck"]].astype(np.float32).values
    meta_sparse = csr_matrix(np.hstack([num_cols, bool_cols]).astype(np.float32))

    # 5. Stack TF-IDF + metadata into one sparse feature matrix
    game_features = hstack([tfidf_matrix, meta_sparse]).tocsr()

    save_npz(out_path / "game_content_features.npz", game_features)
    games.to_parquet(out_path / "games_aligned.parquet", index=False)
    print(f"Saved game_content_features.npz with shape {game_features.shape}")


if __name__ == "__main__":
    build_game_feature_matrix()
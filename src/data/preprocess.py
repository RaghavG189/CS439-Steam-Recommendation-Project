import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.load_data import load_games_and_metadata, load_recommendations_chunked




def run_preprocessing(raw_dir='data/raw', out_dir='data/preprocessed', min_game_reviews=20, min_user_reviews=10):

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    #Run load games and metdata function
    games = load_games_and_metadata(f"{raw_dir}/games.csv", f"{raw_dir}/games_metadata.json")
    games['title'] = games['title'].str.strip()
    games['description'] = games['description'].fillna("").str.strip()
    games['tags_str'] = games['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    games = games.drop(columns=['tags'])

    games_filtered = games[games['user_reviews'] >= min_game_reviews].copy()
    valid_apps = set(games_filtered['app_id'])
    print(f"Kept {len(games_filtered):,} games with >= {min_game_reviews} reviews.")

    #Run load recommendations chunked function
    recs = load_recommendations_chunked(f"{raw_dir}/recommendations.csv", valid_apps=valid_apps)
    recs = recs[recs['hours'] >= 0]
    recs = recs.sort_values('date').drop_duplicates(subset=['user_id', 'app_id'], keep='last')

    user_counts = recs.groupby('user_id')['app_id'].count()
    active_users = user_counts[user_counts >= min_user_reviews].index
    recs = recs[recs['user_id'].isin(active_users)].copy()
    print(f'Retained {len(recs):,} interaction across {len(active_users):,} active users.')

    sign = np.where(recs['is_recommended'], 1.0, -1.0).astype('float32')
    recs['affinity'] = (sign * np.log1p(recs['hours'])).astype('float32')

    recs = recs.sort_values('date').reset_index(drop=True)
    n = len(recs)
    train_end = int(0.80 * n)
    val_end = int(0.90 * n)

    train_df = recs.iloc[:train_end].copy()
    val_df = recs.iloc[train_end:val_end].copy()
    test_df = recs.iloc[val_end:].copy()

    final_apps = set(recs['app_id'].unique())
    games_clean = games_filtered[games_filtered['app_id'].isin(final_apps)].copy()

    train_users = sorted(train_df['user_id'].unique())
    train_apps = sorted(train_df['app_id'].unique())

    mappings = {
        "user_to_idx": {int(u): idx for idx, u in enumerate(train_users)},
        "idx_to_user": {idx: int(u) for idx, u in enumerate(train_users)},
        "app_to_idx": {int(a): idx for idx, a in enumerate(train_apps)},
        "idx_to_app": {idx: int(a) for idx, a in enumerate(train_apps)}
    }

    games_clean.to_parquet(out_path / "games_clean.parquet", index=False)
    train_df.to_parquet(out_path / "train.parquet", index=False)
    val_df.to_parquet(out_path / "val.parquet", index=False)
    test_df.to_parquet(out_path / "test.parquet", index=False)
    with open(out_path / "id_mappingd.json", "w") as f:
        json.dump(mappings, f)
    print(f"Done! Saved splits -> Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

if __name__ == "__main__":
    run_preprocessing()
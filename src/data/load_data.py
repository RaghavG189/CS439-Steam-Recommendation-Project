import pandas as pd

def load_games_and_metadata(games_path='data/raw/games.csv', meta_path = 'data/raw/games_metadata.json'):

    game_dtypes = {
    "app_id": "uint32",
    "title": "str",
    "win": "bool",
    "mac": "bool",
    "linux": "bool",
    "rating": "category",
    "positive_ratio": "uint8",
    "user_reviews": "uint32",
    "price_final": "float32",
    "price_original": "float32",
    "discount": "float32",
    "steam_deck": "bool",
    }

    games_df = pd.read_csv(games_path, dtype=game_dtypes, parse_dates=['date_release'])

    games_meta = pd.read_json(meta_path, lines=True)
    games_meta["app_id"] = games_meta["app_id"].astype("uint32")
    games_df = games_df.merge(games_meta, on="app_id", how="left")

    return games_df


def load_recommendations_chunked(path='data/raw/recommendations.csv', valid_apps=None, chunk_size=500_000):

    rec_dtypes = {
    "app_id": "uint32",
    "helpful": "uint32",
    "funny": "uint32",
    "is_recommended": "bool",
    "hours": "float32",
    "user_id": "uint32",
    "review_id": "uint32",
    }

    use_cols = ['user_id', 'app_id', 'date', 'is_recommended', 'hours']

    chunks = []
    for chunk in pd.read_csv(path, usecols=use_cols, dtype=rec_dtypes, parse_dates=['date'], chunksize=chunk_size):
        if valid_apps is not None:
            chunk = chunk[chunk['app_id'].isin(valid_apps)]
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)
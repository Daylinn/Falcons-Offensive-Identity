import pandas as pd

def load_pbp_season(path: str) -> pd.DataFrame:
    """
    Loads a compressed NFL play-by-play CSV file.
    """
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    return df


def filter_falcons_offense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the PBP data to only include Atlanta Falcons offensive plays.
    """
    return df[df["posteam"] == "ATL"].copy()
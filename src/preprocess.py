import pandas as pd
import numpy as np

class DrivePreprocessor:
    """
    Preprocesses nflverse play-by-play data to create drive-level offensive identity metrics.
    Designed to work with seasons like 2016 & 2023.
    """

    def __init__(self):
        pass

    def _compute_play_metrics(self, df):
        """
        Create run/pass flags + success indicator.
        Uses rush_attempt and pass_attempt directly from nflverse.
        """

        df["rush_attempt"] = pd.to_numeric(df["rush_attempt"], errors="coerce").fillna(0)
        df["pass_attempt"] = pd.to_numeric(df["pass_attempt"], errors="coerce").fillna(0)

        # success flag
        if "success" not in df.columns:
            df["success"] = (df["epa"] > 0).astype(int)

        df["epa"] = pd.to_numeric(df["epa"], errors="coerce").fillna(0)

        df["is_run"] = df["rush_attempt"].astype(int)
        df["is_pass"] = df["pass_attempt"].astype(int)

        return df

    def _aggregate_to_drives(self, df):
        """
        Aggregate play-level metrics into drive-level features.
        """

        # Correct drive result column
        drive_result_col = None
        for col in ["fixed_drive_result", "drive_ended_with_score", "series_result"]:
            if col in df.columns:
                drive_result_col = col
                break

        # Convert yardlines to numeric
        df["drive_start_yard_line"] = pd.to_numeric(df["drive_start_yard_line"], errors="coerce")
        df["drive_end_yard_line"] = pd.to_numeric(df["drive_end_yard_line"], errors="coerce")

        group = df.groupby(["game_id", "drive"])

        drives = group.agg(
            play_count=("play_type", "count"),
            run_pct=("is_run", "mean"),
            pass_pct=("is_pass", "mean"),
            success_rate=("success", "mean"),
            avg_epa=("epa", "mean"),
            start_yardline=("drive_start_yard_line", "first"),
            end_yardline=("drive_end_yard_line", "first"),
            drive_result=(drive_result_col, "first") if drive_result_col else ("play_type", "first")
        ).reset_index()

        # Compute field position gain safely
        drives["field_pos_gain"] = drives["start_yardline"] - drives["end_yardline"]

        return drives

    def transform(self, df):
        """
        Complete preprocessing pipeline.
        """

        df = df.copy()

        df = self._compute_play_metrics(df)
        drives = self._aggregate_to_drives(df)

        return drives
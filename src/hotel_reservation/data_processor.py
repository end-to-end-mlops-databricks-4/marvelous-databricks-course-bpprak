import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservation.config import ProjectConfig


class DataProcessor:
    """Preprocess → split → save to UC Delta (tables only)."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self) -> None:
        """Hotel-reservations preprocessing (moved from notebook):

        - Normalize headers (lowercase, spaces/-/./→ '_')
        - Ensure/align target (case-insensitive fallback)
        - Ensure Id column (string)
        - Coerce numerics + median impute
        - Cast categoricals
        - Drop rows missing target
        - Keep only cat + num + target + Id
        """
        num_features = list(self.config.num_features)
        cat_features = list(self.config.cat_features)
        target = self.config.target

        # --- header normalization & target alignment (from notebook) ---
        _norm = lambda c: c.strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_").lower()
        self.df.columns = [_norm(c) for c in self.df.columns]
        if target not in self.df.columns:  # case-insensitive fallback
            lower_map = {c.lower(): c for c in self.df.columns}
            if target.lower() in lower_map:
                self.df.rename(columns={lower_map[target.lower()]: target}, inplace=True)

        # --- required columns validation ---
        if target not in self.df.columns:
            raise KeyError(f"Required column missing: '{target}'")
        if not any(c in self.df.columns for c in (num_features + cat_features)):
            raise KeyError("No configured features found in input dataframe")

        # --- Id column (from notebook) ---
        if "Id" in self.df.columns:
            self.df["Id"] = self.df["Id"].astype(str)
        elif "booking_id" in self.df.columns:
            self.df["Id"] = self.df["booking_id"].astype(str)
        else:
            self.df["Id"] = pd.Series(range(1, len(self.df) + 1), index=self.df.index).astype(str)

        # --- numerics → numeric + median impute (from notebook) ---
        for col in num_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                med = self.df[col].median()
                if not np.isnan(med):
                    self.df[col] = self.df[col].fillna(med)

        # --- categoricals ---
        for col in cat_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        # --- drop rows with missing target (from notebook) ---
        self.df = self.df.dropna(subset=[target]).copy()

        # --- prune to relevant cols ---
        wanted = cat_features + num_features + [target, "Id"]
        self.df = self.df[[c for c in wanted if c in self.df.columns]].copy()

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        ts = lambda sdf: sdf.withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))
        ts(self.spark.createDataFrame(train_set)).write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )
        ts(self.spark.createDataFrame(test_set)).write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set"
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set"
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

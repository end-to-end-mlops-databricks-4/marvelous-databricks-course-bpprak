"""Data preprocessing module."""

import datetime
import time
from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservation.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df for the Hotel Reservations use case.

        - Validates presence of target/features (raises KeyError if missing)
        - Ensures an Id column exists (string)
        - Coerces numerical features to numeric and imputes missing values with median
        - Casts categorical features to category dtype
        - Drops rows with missing target
        - Keeps only configured columns: cat + num + target + Id
        """
        num_features = list(self.config.num_features)
        cat_features = list(self.config.cat_features)
        target = self.config.target

        # -------------------------
        # Required columns validation (for tests & safety)
        # -------------------------
        if target not in self.df.columns:
            # Empty DF or wrong schema → match test expectation to raise KeyError
            raise KeyError(f"Required column missing: '{target}'")

        if not any(c in self.df.columns for c in (num_features + cat_features)):
            # None of the configured features exist → treat as schema error
            raise KeyError("No configured features found in input dataframe")

        # -------------------------
        # Ensure Id column
        # -------------------------
        if "Id" not in self.df.columns:
            if "booking_id" in self.df.columns:
                self.df["Id"] = self.df["booking_id"].astype(str)
            else:
                self.df["Id"] = pd.Series(range(1, len(self.df) + 1), index=self.df.index).astype(str)
        else:
            self.df["Id"] = self.df["Id"].astype(str)

        # -------------------------
        # Coerce numericals & impute
        # -------------------------
        for col in num_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                med = self.df[col].median()
                if not np.isnan(med):
                    self.df[col] = self.df[col].fillna(med)

        # -------------------------
        # Cast categoricals
        # -------------------------
        for cat_col in cat_features:
            if cat_col in self.df.columns:
                self.df[cat_col] = self.df[cat_col].astype("category")

        # -------------------------
        # Drop rows missing target
        # -------------------------
        self.df = self.df.dropna(subset=[target]).copy()

        # -------------------------
        # Keep only relevant columns that actually exist
        # -------------------------
        wanted = cat_features + num_features + [target, "Id"]
        keep = [c for c in wanted if c in self.df.columns]
        self.df = self.df[keep].copy()

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

"""Unit tests for DataProcessor (Hotel Reservations use case)."""

import pandas as pd
import pytest
from tests.conftest import CATALOG_DIR

from pyspark.sql import SparkSession

# FIX: correct import package name
from hotel_reservation.config import ProjectConfig
from hotel_reservation.data_processor import DataProcessor


def test_data_ingestion(sample_data: pd.DataFrame) -> None:
    """Ensure the sample data has at least one row and one column."""
    assert sample_data.shape[0] > 0
    assert sample_data.shape[1] > 0


def test_dataprocessor_init(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Verify DataProcessor initialization with sample data and spark session."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    assert isinstance(processor.df, pd.DataFrame)
    assert processor.df.equals(sample_data)
    assert isinstance(processor.config, ProjectConfig)
    assert isinstance(processor.spark, SparkSession)


def test_column_transformations(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """
    Hotel dataset version:
    - categoricals → string/category dtype
    - numericals → numeric dtype
    (No house-price specific columns like LotFrontage, MasVnrType, GarageYrBlt.)
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    df = processor.df

    # categoricals are strings/categories
    for c in config.cat_features:
        if c in df.columns:
            assert str(df[c].dtype) in ("string", "object", "category")

    # numericals are numeric
    for n in config.num_features:
        if n in df.columns:
            assert str(df[n].dtype).startswith(("float", "int"))


def test_missing_value_handling(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Numeric columns should have NAs imputed (median fill)."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    df = processor.df

    for n in config.num_features:
        if n in df.columns:
            assert df[n].isna().sum() == 0


def test_column_selection(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Processed DataFrame should contain exactly cat + num + target + Id."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    expected_columns = list(config.cat_features) + list(config.num_features) + [config.target, "Id"]
    assert set(processor.df.columns) == set(expected_columns)


def test_split_data_default_params(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """split_data should produce non-empty train/test with identical schemas."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    train, test = processor.split_data()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) > 0 and len(test) > 0
    assert len(train) + len(test) == len(processor.df)
    assert set(train.columns) == set(test.columns) == set(processor.df.columns)

    # mimic UC tables for downstream tests
    train.to_csv((CATALOG_DIR / "train_set.csv").as_posix(), index=False)  # noqa
    test.to_csv((CATALOG_DIR / "test_set.csv").as_posix(), index=False)  # noqa


def test_preprocess_empty_dataframe(config: ProjectConfig, spark_session: SparkSession) -> None:
    """
    With an empty DataFrame (no columns), selecting required columns should raise KeyError.
    This matches behavior when required columns (e.g., target) are absent.
    """
    processor = DataProcessor(pandas_df=pd.DataFrame([]), config=config, spark=spark_session)
    with pytest.raises(KeyError):
        processor.preprocess()


@pytest.mark.skip(reason="depends on delta tables on Databricks")
def test_save_to_catalog_successful(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """End-to-end save & CDF enable (requires Unity Catalog access on Databricks)."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    train_set, test_set = processor.split_data()
    processor.save_to_catalog(train_set, test_set)
    processor.enable_change_data_feed()

    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.train_set")
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.test_set")

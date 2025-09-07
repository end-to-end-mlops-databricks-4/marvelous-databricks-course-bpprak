"""Dataloader fixture."""

from pathlib import Path
import pandas as pd
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from hotel_reservation import PROJECT_DIR
from hotel_reservation.config import ProjectConfig, Tags
from tests.unit_tests.spark_config import spark_config


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create and return a SparkSession for testing.

    This fixture creates a SparkSession with the specified configuration and returns it for use in tests.
    """
    spark = (
        SparkSession.builder.master(spark_config.master)
        .appName(spark_config.app_name)
        .config("spark.executor.cores", spark_config.spark_executor_cores)
        .config("spark.executor.instances", spark_config.spark_executor_instances)
        .config("spark.sql.shuffle.partitions", spark_config.spark_sql_shuffle_partitions)
        .config("spark.driver.bindAddress", spark_config.spark_driver_bindAddress)
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    Reads the project configuration from YAML and returns a ProjectConfig instance.
    """
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    return ProjectConfig.from_yaml(config_file_path.as_posix(), env="dev")


@pytest.fixture(scope="function")
def sample_data(config: ProjectConfig, spark_session: SparkSession) -> pd.DataFrame:
    """Create a sample DataFrame from a Hotel Reservations CSV.

    Tries common hotel sample locations; falls back to project data folder.
    Returns a Pandas DataFrame.
    """
    # Candidate locations for hotel sample data
    candidates = [
        PROJECT_DIR / "tests" / "test_data" / "hotel" / "sample.csv",
        PROJECT_DIR / "tests" / "test_data" / "hotel_sample.csv",
        PROJECT_DIR / "tests" / "test_data" / "data.csv",
        PROJECT_DIR / "data" / "data.csv",
    ]

    csv_path: Path | None = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Hotel sample CSV not found. Expected one of: "
            + ", ".join(p.as_posix() for p in candidates)
        )

    logger.info(f"Loading hotel sample CSV: {csv_path.as_posix()}")
    sample = pd.read_csv(csv_path.as_posix())

    # NOTE: Do not coerce/normalize here; DataProcessor.preprocess() handles that.
    return sample


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Provide a Tags instance for the test session."""
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")

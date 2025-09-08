# Databricks notebook source

# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

# Use your existing package/modules
from hotel_reservation.config import ProjectConfig
from hotel_reservation.data_processor import DataProcessor

# Load config (already adapted for hotel reservations: cat_features/num_features/target=booking_status)
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Load the Hotel Reservations dataset
spark = SparkSession.builder.getOrCreate()

filepath = "../data/data.csv"  # point to your Kaggle CSV (rename/move as needed)

# Load the data
df = pd.read_csv(filepath)


# ---- Minimal header normalization & Id handling (keeps your modules unchanged) ----
def _normalize(col: str) -> str:
    return col.strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_").lower()


df.columns = [_normalize(c) for c in df.columns]

# Ensure target column name from YAML exists (case-insensitive fallback handled by normalize above)
target = config.target
if target not in df.columns:
    lower_map = {c.lower(): c for c in df.columns}
    if target.lower() in lower_map:
        df.rename(columns={lower_map[target.lower()]: target}, inplace=True)
    else:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

# Ensure an Id column (your DataProcessor expects it)
if "booking_id" in df.columns:
    df["Id"] = df["booking_id"].astype(str)
else:
    df["Id"] = (pd.Series(range(1, len(df) + 1))).astype(str)

# Optional: basic dtype coercion for robustness (uses YAML lists)
for c in config.cat_features:
    if c in df.columns:
        df[c] = df[c].astype("string")
for n in config.num_features:
    if n in df.columns:
        df[n] = pd.to_numeric(df[n], errors="coerce")

# Drop rows missing the target (safety)
df = df.dropna(subset=[target]).copy()

# COMMAND ----------
# Pipeline using your existing DataProcessor API

data_processor = DataProcessor(df, config, spark)

# Preprocess the data (your module’s method; now safe for hotel schema)
data_processor.preprocess()

logger.info("Data preprocessing is completed.")

# COMMAND ----------
# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()

# COMMAND ----------

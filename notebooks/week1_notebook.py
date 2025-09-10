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

from hotel_reservation.config import ProjectConfig
from hotel_reservation.data_processor import DataProcessor

# Load config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Load the Hotel Reservations dataset
spark = SparkSession.builder.getOrCreate()
filepath = "../data/data.csv"
df = pd.read_csv(filepath)

# COMMAND ----------
# Pipeline using DataProcessor 
processor = DataProcessor(df, config, spark)
processor.preprocess()
X_train, X_test = processor.split_data()

logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

logger.info("Saving data to catalog")
processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
#logger.info("Enable change data feed")
#processor.enable_change_data_feed()



# COMMAND ----------

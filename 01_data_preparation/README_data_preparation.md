# Python Data Preparation and Feature Engineering 2023

## Overview

- **Python Script** — Contains all data preparation steps and engineered features for the dataset.

- **Raw Data Parquet File** — Includes data from academic years 1213–2223, sourced from the Knime workflow.

- **Featured Parquet File (by Year)** — Data with added features and a labeled target variable based on **YEAR**.

- **Featured Parquet File (by Term)** — Data with added features and a labeled target variable based on **TERM**.

- **Deduplicated Parquet File (by Year)** — Data with added features and a labeled target based on **YEAR**, using a randomly selected single record per `edw_pers_id` to avoid overfitting on duplicate entries.

- **Readme Documentation** — Provides details on aggregation methods and their purposes.

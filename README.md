# Enrollment Prediction and Forecasting System (UIC)

End-to-end machine learning and forecasting system for student enrollment and retention at the University of Illinois Chicago (UIC). This project combines classification modeling and time series forecasting to identify at-risk students and generate long-term enrollment projections across academic units.

---

## Overview

This project addresses two core institutional challenges:

1. **Identifying students at risk of discontinuing enrollment**
2. **Forecasting future enrollment for planning and resource allocation**

The solution integrates:
- Machine learning classification (Random Forest / Tree Ensemble)
- Time series forecasting (ARIMA, Exponential Smoothing)
- Feature engineering from large-scale academic data
- Multi-level outputs (student, department, college)

> Built as a full lifecycle system from data ingestion to executive-level insights. :contentReference[oaicite:0]{index=0}

---

## Key Results

- ~82% classification accuracy for student continuation prediction
- 600K+ student records modeled across 10 years
- Forecasting across 14 colleges
- 10-year enrollment projections generated
- Delivered to institutional leadership for planning use cases :contentReference[oaicite:1]{index=1}

---

## System Architecture

This project is structured as a full ML pipeline:

01_data_preparation
02_feature_engineering
03_modeling_tree_ensemble
04_predictions_ay2024
05_projections_10yr


### Pipeline Breakdown

**Data Preparation**
- Raw data sourced from Oracle EDW
- Cleaned, standardized, and stored in Parquet format
- Temporal structuring for student-level tracking

**Feature Engineering**
- Time-based progression features (terms, academic years)
- Student profile and academic indicators
- Target variable design (multiple labeling strategies evaluated)

**Modeling (Classification)**
- H2O Random Forest (selected model)
- CatBoost (evaluated)
- SMOTE for class balancing
- Cross-validation and confusion matrix evaluation

**Forecasting (Time Series)**
- ARIMA models for enrollment projection
- Exponential smoothing for trend-based forecasts
- Separate modeling for incoming and continuing students

**Outputs**
- Student-level risk predictions
- College-level aggregated insights
- 10-year enrollment forecasts

---

## Modeling Approach

### Classification

- Model Type: Tree Ensemble (H2O Random Forest)
- Target: Student continuation vs discontinuation
- Validation: Cross-validation + holdout testing

**Performance**
- Accuracy: ~94.8%
- F1 Score: 0.9676
- AUC: 0.9847 :contentReference[oaicite:2]{index=2}

### Forecasting

- ARIMA for structured time series modeling
- Exponential smoothing for trend capture
- Forecast horizon: 10 years

---

## Data

- Source: University of Illinois Enterprise Data Warehouse (Oracle)
- Time Range: Fall 2013 – Summer 2023
- Scale:
  - ~600K+ total records
  - Undergraduate + Graduate populations
- Structure:
  - One row per student per term (Census snapshot)

---

## Key Insights

- Time-in-program is the strongest predictor of retention
- Early-term students have the highest discontinuation risk
- Graduation and continuation behaviors are statistically similar
- Forecasting requires segmentation by student population

---

## Business Impact

This system enables:

- Proactive identification of at-risk students
- Improved advising and intervention strategies
- Long-term enrollment planning
- Resource allocation across colleges and departments

Outputs support decision-making at:
- Student level (advisors)
- College level (deans)
- Institutional level (leadership)

---

## Tech Stack

- Python (pandas, modeling, forecasting)
- H2O (Random Forest)
- KNIME (data pipelines, workflow orchestration)
- ARIMA (time series modeling)
- Oracle EDW (data source)

---

## Repository Structure

models/ # ML + forecasting models
workflow/ # KNIME workflows
outputs/ # predictions and forecasts
data/ # processed datasets
docs/ # supporting documentation


---

## Notes

- Multiple target labeling strategies were tested before selecting final approach
- Data leakage was explicitly addressed during feature selection
- Aggregation strategies were evaluated to prevent overfitting
- Designed for scalability and institutional deployment

---

## Author

Tayler Erbe  
Senior Data Scientist  
University of Illinois System

---

## Links

- Case Study: (add your portfolio link)
- GitHub Repo: (this repo)

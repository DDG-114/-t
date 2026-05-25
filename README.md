# Datang Electricity Price Forecasting

This repository contains the code and documentation for the day-ahead electricity price forecasting experiments.

The raw and processed datasets are intentionally excluded from Git. Place local data files under each project's `data/raw/` directory before running the pipelines.

Main project:

- `policy_aware_tcn_project/`: policy-aware TCN model, training, evaluation, and tests.
- `dayahead_epf_agent_project/`: baseline feature engineering, LightGBM baseline, and earlier TCN smoke workflow.


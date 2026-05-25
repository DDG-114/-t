# Policy-Aware TCN Project

This directory is the standalone deep-model project separated from the LightGBM baseline project.

It implements the report2 recommendation:

```text
supply-demand prior price model + compact TCN residual correction
+ business-aligned loss + year/policy-aware post-processing
```

The baseline project remains in:

```text
/home/kaga/Desktop/datang/dayahead_epf_agent_project
```

This project lives in:

```text
/home/kaga/Desktop/datang/policy_aware_tcn_project
```

## Models

The implemented model is `PolicyAwareTCN`:

- supply-demand prior: a train-split-only piecewise-linear Ridge model maps
  market covariates such as net load, bidding space, renewable share, and
  supply-demand margin to `sd_prior_price`;
- fundamental uncertainty features: same-slot rolling quantile proxies
  (`q10/q50/q90` and spreads) are constructed for load, renewable, bidding,
  generation, tie-line, residual-load, and supply-margin variables using only
  previous days, then used by both the prior model and the TCN future branch;
- historical branch: previous 14 days, `14 x 96 = 1344` time steps;
- future branch: target-day known covariates for all 96 delivery slots;
- static policy branch: `policy_regime`, `price_lower_bound`, `price_upper_bound`, `is_zero_floor_regime`;
- TCN backbone: causal dilated residual blocks with dilation `[1, 2, 4, 8, 16, 32]`;
- head: direct 96-step residual price prediction over the supply-demand prior
  price anchor;
- uncertainty: optional `log_sigma` head for heteroscedastic auxiliary loss.

Default parameter count is about 201k after missing-indicator channels are appended.

The project also includes `state_gbm`, a stronger state-aware LightGBM baseline:

- adds day-ahead-safe boundary-state history features for floor/high/cap prices;
- trains a weighted MAE LightGBM regressor;
- trains separate floor-price, high-price, and cap-price classifiers;
- selects state override thresholds on the validation split;
- refits the final model through the validation end before evaluating the test
  split, following the rolling-calibration style used in EPF literature.

## Loss

Training uses:

```text
weighted relative MAE + ramp regularization + heteroscedastic NLL auxiliary loss
```

Model selection and early stopping use validation `mean_daily_accuracy`, so the saved checkpoint follows the same primary business metric used in reports.

Missing target labels are handled with `y_mask`; they do not enter the loss or evaluation metrics.

## Policy Handling

The network does not hard-code the 2025 lower bound. Predictions are clipped after inference:

```text
2025: [40, 1000]
2026: [0, 1000]
```

The config also enables conservative low-weight synthetic zero-floor samples for 2026-style lower-tail adaptation.

## Run

Use the virtual environment already created in the baseline project:

```bash
cd /home/kaga/Desktop/datang/policy_aware_tcn_project
../dayahead_epf_agent_project/.venv/bin/python scripts/02_train_policy_tcn.py --config config/default.yaml
../dayahead_epf_agent_project/.venv/bin/python scripts/03_evaluate_policy_tcn.py --config config/default.yaml
```

Train and evaluate the state-aware GBM:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/04_train_evaluate_state_gbm.py --config config/default.yaml
```

Build an optional weather-augmented feature table from Open-Meteo historical
forecast runs:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/07_build_weather_features.py \
  --config config/default.yaml \
  --input-features data/processed/features.csv \
  --output-features outputs/weather_experiment/features_weather.csv
```

Rebuild features if needed:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/01_build_features.py --config config/default.yaml
```

Run tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../dayahead_epf_agent_project/.venv/bin/python -m pytest -q
```

## Outputs

```text
outputs/models/policy_aware_tcn.pt
outputs/models/policy_aware_tcn_report.json
outputs/models/supply_demand_prior.pkl
outputs/models/supply_demand_prior_report.json
outputs/predictions/policy_aware_tcn_predictions.csv
outputs/reports/policy_aware_tcn_summary.json
outputs/predictions/policy_aware_tcn_future_24h_predictions.csv
outputs/reports/policy_aware_tcn_future_24h_summary.json
```

State-aware GBM outputs:

```text
outputs/models/state_gbm.pkl
outputs/models/state_gbm_report.json
outputs/predictions/state_gbm_predictions.csv
outputs/reports/state_gbm_summary.json
```

Weather-enhanced experiments use the same state-aware GBM script after pointing
`paths.processed_features` at the weather-augmented feature CSV.
Lagged weather forecast-error features can be added on top of that table:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/08_add_weather_error_history.py \
  --config config/gpu.yaml \
  --input-features outputs/weather_experiment/features_weather.csv \
  --output-features outputs/weather_error_experiment/features_weather_error.csv \
  --forecast-lead-day 1 \
  --windows-days 1 3 7
```

The quantile features are leakage-safe proxies because the current raw dataset
contains fundamental point forecasts but not the realized load/renewable series
needed to postprocess forecast errors. If realized fundamentals become available,
this module can be upgraded to the QR/HS probabilistic-input route described in
the literature.

## Current Accuracy Ceiling

On the December 2025 test split, the strongest verified internal-data model so
far is the state-aware GBM with Open-Meteo forecast features plus lagged
weather forecast-error history, trained from 2024-04-01:

```text
accuracy: 0.7692
MAE: 82.33
RMSE: 166.32
cap_normalized_accuracy: 0.9177
```

Without weather, the same state-aware GBM reached `accuracy=0.7444`. Weather
forecast-run features reached `accuracy=0.7498`, and extending the training
window to start at 2024-04-01 reached `accuracy=0.7654`. Adding lagged weather
forecast-error history lifted the verified result to `accuracy=0.7692`, mainly
by raising predictions in the 150+ price regions. It still does not close the
85% gap.

The main remaining bottleneck is the high-price boundary. A diagnostic oracle
showed that perfect floor-state correction would lift accuracy to about 0.82,
while perfect floor plus perfect 900+ cap-state correction would be needed to
reach about 0.85. With the current dataset, the cap-state classifier remains
weak because the raw data contains fundamental forecasts but not realized
fundamental observations. The next data priority is to add realized
load/renewable/generation series, or an official market scarcity signal, so
cap-price scarcity states can be identified before the delivery day.

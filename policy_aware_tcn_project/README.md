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

A residual calibration layer can be trained after the state-aware GBM:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/09_train_evaluate_residual_calibrator.py \
  --config config/residual_calibrator_weather_error_q2.yaml
```

For sequential backtests where earlier realized days in the test month are
available, apply the leakage-safe online residual calibrator:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/10_apply_online_residual_calibrator.py \
  --config config/online_residual_best.yaml \
  --predictions outputs/residual_calibrated_train2024q2/predictions/state_gbm_residual_predictions.csv \
  --prefix online_residual_best
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

External scarcity or realized-fundamental data can be merged into an existing
feature table without rebuilding all weather features:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/15_audit_external_signal_file.py \
  --config config/external_scarcity_template.yaml

../dayahead_epf_agent_project/.venv/bin/python scripts/12_add_external_signals.py \
  --config config/external_scarcity_template.yaml \
  --input-features outputs/prevday_curve_2025_priority/features_prevday_curve_weather_error.csv \
  --output-features outputs/external_scarcity/features_external_scarcity.csv
```

Use `direct_columns` only for signals known before the day-ahead forecast, and
put realized load/renewable/generation/reserve-margin columns under
`lagged_actual_columns`; those are exposed only as previous-day lag and rolling
features.

After adding new scarcity columns, check whether they actually improve high-price
ranking before running a full training cycle:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/13_diagnose_high_price_regime.py \
  --config config/residual_calibrator_weather_error_q2.yaml \
  --features outputs/external_scarcity/features_external_scarcity.csv \
  --predictions outputs/residual_calibrated_train2024q2/predictions/gbm_residual_anchor_all_predictions.csv \
  --target-threshold 500 \
  --top-k 120
```

Use the gap analyzer to inspect where a candidate prediction file still misses
the 85% target:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/14_analyze_accuracy_gap.py \
  --predictions outputs/online_residual_best/predictions/online_residual_best_predictions.csv
```

Online residual parameters can be re-selected on a validation window, but this
has not beaten the current hand-selected online best on 2025-12:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/16_tune_online_residual_calibrator.py \
  --config config/online_residual_best.yaml \
  --predictions outputs/residual_calibrated_train2024q2/predictions/gbm_residual_anchor_all_predictions.csv \
  --prefix online_residual_tuned_valid100
```

GBM is now archived as the strongest classical baseline; see
`docs/gbm_archive.md`. The active deep-learning route can use GPU training with
weather and weather-error features:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/02_train_policy_tcn.py \
  --config config/deep_weather_highcap.yaml
../dayahead_epf_agent_project/.venv/bin/python scripts/03_evaluate_policy_tcn.py \
  --config config/deep_weather_highcap.yaml
```

The first GPU deep runs are not yet competitive with the GBM baseline; the best
deep test accuracy so far is `0.6707`.

## Current Accuracy Ceiling

On the December 2025 test split, the strongest verified internal-data result so
far is the residual-calibrated state-aware GBM plus online residual adaptation.
The online layer uses only earlier realized days in the December 2025 sequence:

```text
accuracy: 0.7857
MAE: 80.43
RMSE: 166.83
cap_normalized_accuracy: 0.9195
```

The strongest pre-online base model is the state-aware GBM with Open-Meteo
forecast features, lagged weather forecast-error history, and a
validation-selected residual calibration layer, trained from 2024-04-01:
`accuracy=0.7711`, `MAE=90.01`, `RMSE=177.03`.

Without weather, the same state-aware GBM reached `accuracy=0.7444`. Weather
forecast-run features reached `accuracy=0.7498`, and extending the training
window to start at 2024-04-01 reached `accuracy=0.7654`. Adding lagged weather
forecast-error history lifted the verified result to `accuracy=0.7692`, mainly
by raising predictions in the 150+ price regions. A residual calibration layer
selected on the validation split lifted the result to `accuracy=0.7711`. Online
residual adaptation lifted it further to `accuracy=0.7857`. It still does not
close the 85% gap.

The main remaining bottleneck is the high-price boundary. A diagnostic oracle
showed that perfect floor-state correction would lift accuracy to about 0.82,
while perfect floor plus perfect 900+ cap-state correction would be needed to
reach about 0.85. With the current dataset, the cap-state classifier remains
weak because the raw data contains fundamental forecasts but not realized
fundamental observations. The next data priority is to add realized
load/renewable/generation series, or an official market scarcity signal, so
cap-price scarcity states can be identified before the delivery day.

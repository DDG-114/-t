# Accuracy Experiment Log

This log records no-leakage experiments aimed at the project accuracy metric:

```text
accuracy = 1 - mean(abs(y_pred - y_true) / max(abs(y_true), 40))
```

The fixed test period is 2025-12-01 through 2025-12-31 unless noted otherwise.

## Current Best Reproducible Test Result

- Config: `config/online_residual_best.yaml`
- Command:
  ```bash
  ../dayahead_epf_agent_project/.venv/bin/python scripts/10_apply_online_residual_calibrator.py \
    --config config/online_residual_best.yaml \
    --predictions outputs/residual_calibrated_train2024q2/predictions/state_gbm_residual_predictions.csv \
    --prefix online_residual_best
  ```
- Output: `outputs/online_residual_best/reports/online_residual_best_summary.json`
- Test accuracy: `0.7857284442214671`
- MAE: `80.43487190860117`
- RMSE: `166.8320110819527`
- Pass days at 85% daily accuracy: `9 / 31`

The online residual calibrator uses only earlier days inside the 2025-12
evaluation stream. Its current validation-selected setting waits for ten earlier
realized days and then corrects each day from the previous five days of observed
prediction errors, grouped by prediction bins, so it is compatible with a
rolling day-ahead deployment where earlier realized prices are available.

The online residual settings were selected on the 2025-10 through 2025-11
validation stream. In a small grid over `window_days`, `min_history_days`, and
`shrink`, the best validation accuracy was `0.8187824307784491` with
`window_days=5`, `min_history_days=10`, and `shrink=0.5`; the corresponding
2025-12 test accuracy is `0.7857284442214671`.

The best pre-online base model remains:

- Config: `config/residual_calibrator_weather_error_q2.yaml`
- Output: `outputs/residual_calibrated_train2024q2/reports/state_gbm_residual_summary.json`
- Test accuracy: `0.7711041307390922`

## 2025-Priority Experiment

- Config: `config/residual_calibrator_2025_priority.yaml`
- Train: 2025-01-01 through 2025-10-31
- Validation: 2025-11-01 through 2025-11-30
- Test: 2025-12-01 through 2025-12-31
- StateGBM test accuracy: `0.7605732816532624`
- Residual-calibrated test accuracy: `0.7573985596079574`

This confirms that prioritizing 2025 data improves recency alignment on 2025-11,
but it does not by itself solve the 2025-12 high-price regime shift.

## Literature-Inspired Net Bidding Space Diagnostic

A related EPF idea is to predict day-ahead clearing prices with net spot bidding
space after accounting for medium/long-term market effects. The local dataset
contains `竞价空间`, but it does not contain the contract/locked-energy fields
needed to compute true net spot bidding space. A leakage-safe proxy was tested:

```text
net_bidding_space_proxy = 竞价空间 - alpha * 统一负荷预测
```

where `alpha` was estimated from historical same-slot floor/high/cap persistence.
With 2025-priority training, this proxy reached only about `0.7559` test
accuracy, below the `0.7606` 2025-priority baseline and below the current best.

The result suggests that true medium/long-term contract data or official net
spot bidding-space data is needed; a price-history proxy is not sufficient.

## 2025 Rolling-Week Diagnostic

- Train/validation/test were rolled by week inside 2025.
- Each week used only data available before the predicted week.
- Combined 2025-12 accuracy: about `0.7053`.

The rolling approach overreacted to recent periods and raised too many normal or
low-price slots. It is not adopted as the default path.

## Negative / Ablation Results

- 2025-priority quantile LightGBM:
  - Trained quantile regressors at q50/q60/q70/q80/q90 and blended upper
    quantiles into suspected high-price slots.
  - Best validation-selected test accuracy was about `0.7546`.
  - Test-oracle capacity within this blend family was about `0.7619`.
  - Result: upper-quantile regressors still fail to identify enough December
    high/cap slots.
- Previous-day price-curve features:
  - Added neighboring-slot prices from the complete previous-day curve plus
    previous-day curve mean/max/min/std and floor/high/cap ratios.
  - 2025-priority validation reached about `0.8771`, but 2025-12 test accuracy
    was only about `0.7570`.
  - Result: useful for floor-price persistence, but not enough for December
    high/cap regime transfer.
- China holiday / adjusted-workday calendar features:
  - Added legal holiday, makeup-workday, workday/rest-day, and distance-to-holiday features.
  - 2025-priority validation improved to about `0.8790`, but 2025-12 test accuracy was only about `0.7579`.
  - Result: the 2025-11 calendar relationship did not transfer to December.
- 2025-priority scarcity rule postprocessing:
  - Validation-selected rule used high net load and low renewable forecast to
    lift suspected scarcity slots.
  - Test accuracy improved only from `0.7605732816532624` to about `0.7639`.
  - Result: useful diagnostic, but still lower than the current best.
- 2025-priority prediction-bin calibration:
  - Validation-selected threshold calibration improved 2025-11 slightly.
  - Test accuracy decreased to about `0.7587`.
  - Result: 2025-11 calibration relationships do not transfer to 2025-12.
- Scarcity rolling-state features:
  - Config: `config/residual_calibrator_scarcity_q2.yaml`
  - Residual-calibrated test accuracy: `0.7472349106265357`
  - Result: lower than the current best.
- Four-state floor/normal/high/cap mixture-of-experts diagnostic:
  - Used 2025-priority training and a multiclass regime classifier.
  - Validation-selected test accuracy was about `0.7335`.
  - Result: high/cap December slots were still mostly classified as normal.
- November-only validation after training through 2025-10:
  - Config: `config/residual_calibrator_weather_error_novvalid.yaml`
  - Residual-calibrated test accuracy: `0.6770950754961472`
  - Result: validation window is too small/unstable for December.
- Simple no-leakage fusion between old-window and 2025-priority StateGBM:
  - Validation-selected fusion test accuracy: about `0.7456`
  - Test-oracle fusion capacity: about `0.7643`
  - Result: not enough complementarity to reach the current best.
- ExtraTrees diagnostic:
  - Best tested state-overridden accuracy was below `0.75`.
  - Result: still underpredicts high/cap price periods.

## Main Bottleneck

The December 2025 high-price and cap-price periods remain the bottleneck. Current
features can classify floor-price periods well, but high and cap probabilities
are not strong enough to lift predictions without damaging normal periods. The
best next improvement likely requires new leakage-safe external signals that
describe realized or forecast scarcity more directly, such as actual load,
renewable output, available capacity, reserve margin, outage/maintenance, or
official market disclosure data.

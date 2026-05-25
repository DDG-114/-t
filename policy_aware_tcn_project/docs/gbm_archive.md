# GBM Route Archive

This document archives the state-aware GBM route as a reproducible baseline.
The active improvement route has moved to deep learning.

## Best Verified GBM Result

- Split: train `2024-04-01` through `2025-09-30`, validation `2025-10-01`
  through `2025-11-30`, test `2025-12-01` through `2025-12-31`
- Base command:
  ```bash
  ../dayahead_epf_agent_project/.venv/bin/python scripts/09_train_evaluate_residual_calibrator.py \
    --config config/residual_calibrator_weather_error_q2.yaml
  ```
- Online residual command:
  ```bash
  ../dayahead_epf_agent_project/.venv/bin/python scripts/10_apply_online_residual_calibrator.py \
    --config config/online_residual_best.yaml \
    --predictions outputs/residual_calibrated_train2024q2/predictions/state_gbm_residual_predictions.csv \
    --prefix online_residual_best
  ```
- Best test accuracy: `0.7857284442214671`
- MAE: `80.43487190860117`
- RMSE: `166.8320110819527`
- Pass days at 85% daily accuracy: `9 / 31`

## Bottleneck

The GBM route already handles floor-price slots well. On the December 2025 test
split, true floor slots reached about `0.94` point accuracy after online
residual adaptation. The unresolved issue is high-price and cap-price recall:
the test month contains `426` slots with true price at or above `500`, but the
best GBM-plus-online route predicts only `14` slots at or above `500`.

Static high-price lift rules selected on validation did not transfer strongly.
The best diagnostic test-only high-price rule lifted the online result only to
about `0.7860`, which is too small to justify adding another GBM postprocessor.

## Negative GBM Follow-up

A residual-calibrator refit experiment used validation data through
`2025-11-30` to refit the residual layer after selecting parameters on
validation:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/04_train_evaluate_state_gbm.py \
  --config config/residual_calibrator_refit_q2.yaml
../dayahead_epf_agent_project/.venv/bin/python scripts/09_train_evaluate_residual_calibrator.py \
  --config config/residual_calibrator_refit_q2.yaml
```

It reached only `0.7646436863432745` test accuracy, below both the
pre-online residual-calibrated GBM and the online residual best.

## Status

GBM code, configs, and reports remain useful for comparison and ablation, but
they are no longer the main optimization path. The next route is a GPU-trained
deep model with weather inputs, supply-demand prior anchoring, and validation
selected high-price correction.

The first post-archive deep route now uses GBM as an external anchor rather than
continuing GBM tuning. `config/deep_gbm_anchor_conservative.yaml` trains a TCN
residual model on GPU and keeps a validation-selected residual blend gate so the
deep output can fall back to the GBM-residual anchor when the learned residual
does not clear a small validation-improvement threshold.

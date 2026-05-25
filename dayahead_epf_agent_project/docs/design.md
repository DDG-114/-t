# Design Notes

## Why keep slot-wise LightGBM?

This project predicts 96 day-ahead electricity prices per day. Each 15-minute delivery interval can have different market behaviour. Midnight, solar noon, evening peak, and late-night intervals are driven by different combinations of load, renewable generation, bidding space, pumped storage, and interconnector schedule.

Therefore, the first engineering model uses 96 independent LightGBM regressors. This is not the newest research model, but it is a strong, robust, explainable, and deployable method for structured electricity-market data. It remains the comparison baseline for any new deep model.

## Policy-Aware TCN Architecture

Report2 recommends a compact TCN as the first deep learning architecture because this dataset has limited supervised days, strong 15-minute local structure, and known future market covariates. The implemented architecture is not a placeholder:

- historical branch: causal dilated residual TCN over the previous 14 days;
- future branch: temporal MLP encoder for target-day load, renewable, bidding-space, pumped-storage, interconnector, market-derived, calendar, and lag/rolling context features;
- policy branch: static conditioning on `policy_regime`, `price_lower_bound`, `price_upper_bound`, and `is_zero_floor_regime`;
- fusion: gated context/future/policy fusion for each of the 96 delivery slots;
- output: direct 96-point residual correction over yesterday's same-slot price anchor;
- auxiliary head: optional log-sigma head for heteroscedastic training.

The model is trained with a business-aligned objective:

```text
weighted_relative_mae + lambda_ramp * ramp_error + nll_weight * heteroscedastic_nll
```

The relative denominator uses the configured `metrics.price_floor`, matching daily accuracy. Point weights emphasize high-price, low-price, and boundary slots. Ramp regularization discourages unrealistic 15-minute jumps.

The network output does not hard-code the 2025 floor. Predictions are clipped after inference using year/policy bounds:

```text
2025 -> [40, 1000]
2026 -> [0, 1000]
```

For the 2026 zero-floor regime, the training code can add conservative low-weight synthetic samples only from slots that already look like low-price candidates: low observed prices plus high renewable share, low net load, or pumped-storage charging context.

## Model Input

For each target date and slot, features include:

- historical same-slot price lags;
- rolling same-slot statistics;
- forecasted load, renewable output, generation output, pumped storage, bidding space, and interconnector schedule;
- market-aware derived features such as net load and supply-demand margin;
- calendar and cyclic slot features.

## Metric Definition

Daily accuracy is currently implemented as:

```text
daily_accuracy = 1 - mean(abs(pred - actual) / max(abs(actual), price_floor))
```

This is a practical modified formula because the dataset contains zero-price points. If the formal project contract or teacher gives a different formula, replace the implementation in `src/epf/metrics.py`.

## Next Research Extensions

After the first version is stable:

1. compare global LightGBM and XGBoost;
2. add LSTM sequence-to-sequence as the second deep model;
3. add PatchTST-lite or TFT-lite only after the TCN/LSTM baselines are stable;
4. add ensemble prediction;
5. evaluate low-price, high-price, and spike-price periods separately;
6. add conformal probabilistic forecast intervals;
7. connect forecast errors to bidding strategy and system operation risk analysis.

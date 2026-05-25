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

## GBM-Anchor Deep Residual Route

After archiving direct GBM tuning, the next deep-learning structure uses the
residual-calibrated GBM prediction as an external anchor and trains the TCN only
to adjust that anchor. The anchor rows are generated with:

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/11_generate_residual_anchor_predictions.py \
  --config config/residual_calibrator_weather_error_q2.yaml \
  --prefix gbm_residual_anchor
```

- `config/deep_gbm_anchor_residual.yaml`: GPU-trained TCN residual model with
  validation-selected residual blending and high-price correction. Validation
  accuracy improved from the anchor's `0.8176921731389057` to
  `0.8206149944565261`, but the 2025-12 test accuracy dropped to
  `0.769736984140805`, below the anchor.
- `config/deep_gbm_anchor_conservative.yaml`: same model, but the residual blend
  requires at least `0.003` validation accuracy improvement before changing the
  anchor. It therefore falls back to the GBM-residual anchor and reaches
  `0.7711041306166069` on the 2025-12 test split.
- Applying the existing online residual calibrator to the conservative
  GBM-anchor deep output reaches `0.7857284437537457`, effectively the same as
  the archived online GBM best.

Result: the deep residual route is now structurally wired and GPU-runnable, but
the TCN residual itself has not yet improved December 2025 high-price recall.
On the 2025-12 test split there are `426` true slots at or above `500`; the
anchor predicts only `7`, and the raw deep residual model predicts only `6`.
The route is useful as a safe deep-learning scaffold, but it has not solved the
high/cap boundary bottleneck.

## Spike-Aware Online Diagnostics

Electricity-price forecasting literature commonly treats price spikes as a
separate regime rather than as ordinary regression noise: review papers by Weron
and later EPF surveys discuss variance-stabilizing transforms, regime-switching
models, and separate spike handling; more recent deep EPF work often combines
sequence models with exogenous variables and rolling/online adaptation. The
local experiments below follow that direction but keep the no-leakage day-ahead
contract.

- Added `online_residual_calibrator.spike_residual` as an optional online
  residual branch for high/cap-probability slots. It can group recent residuals
  by state, spike-state, prediction bin, or hour and use a configurable residual
  quantile.
- Default spike branch:
  ```bash
  ../dayahead_epf_agent_project/.venv/bin/python scripts/10_apply_online_residual_calibrator.py \
    --config config/online_residual_spike.yaml \
    --predictions outputs/residual_calibrated_train2024q2/predictions/state_gbm_residual_predictions.csv \
    --prefix online_residual_spike
  ```
  reached test accuracy `0.7848057383085196`, below the current online best.
- Manual validation/test diagnostics found some test-only gains from stronger
  spike residuals, for example hour-grouped high-probability residuals reached
  about `0.7882` on 2025-12, but their 2025-10/11 validation accuracy fell
  materially. Because the validation relationship does not transfer, this is not
  adopted as the selected result.
- A test-only upper-bound diagnostic using existing scores showed that fixed
  high-price lifting based on `high_probability`, `cap_probability`, or
  `y_pred` can only reach about `0.7884`. If the true high-price mask
  (`y_true >= 500`) were known but high slots were set to a constant price, the
  score would be about `0.8174`; if high slots were perfectly predicted, the
  score would be about `0.8484`.

Result: the current feature set has enough information to rank high-price slots
roughly, but not enough to estimate their magnitude accurately. Reaching `0.85`
is unlikely from threshold lifting or online residual tuning alone; it likely
requires a stronger high-price magnitude signal such as actual/forecast reserve
margin, outage/maintenance, available capacity, real-time load/renewable error,
or official scarcity/market disclosure features.

## High-Price Magnitude Expert Diagnostic

Following the spike-regime literature, a separate high-price magnitude expert
was tested on top of the GBM-residual anchor. The expert used leakage-safe
fundamental forecasts, price history, anchor predictions, and state
probabilities, with training masks based on high/cap probability or elevated
anchor price. Ridge/Huber variants were tested for raw price, residual, and
log-price targets, with the blend/gate chosen on the 2025-10/11 validation
period.

Best validation-selected variants transferred poorly to 2025-12: the strongest
simple magnitude experts reached only about `0.7711` test accuracy, below the
online residual best `0.7857284442214671`. Slower tree-based expert grids were
also attempted but were not worth adopting because the lightweight diagnostic
already showed poor validation-to-test transfer and the earlier oracle analysis
put fixed-score lifting below `0.789`.

Result: a separate high-price magnitude model using the current local feature
table is still not enough. The next useful route is to add new no-leakage
scarcity signals rather than continuing to tune models on the same inputs.

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

- Deep learning route after GBM archive:
  - Added optional weather/weather-error features to the TCN sequence inputs
    and a validation-selected high-price correction layer.
  - `config/deep_weather_highcap.yaml`: GPU-trained weather TCN with
    supply-demand-prior anchor, synthetic zero-floor samples, floor correction,
    and high-price correction. Validation accuracy was `0.6706`; 2025-12 test
    accuracy was `0.67069897689749`.
  - `config/deep_weather_highcap_v2.yaml`: disabled synthetic zero-floor samples
    and used previous-day anchor. Test accuracy was `0.6213359746179685`.
  - `config/deep_weather_highcap_v3.yaml`: additionally disabled floor
    correction. Test accuracy was `0.4428512776960952`.
  - Result: the current TCN is runnable on GPU and now consumes weather inputs,
    but it is not yet competitive with the GBM baseline. The main issue is an
    unstable price anchor / floor-high tradeoff; the next deep structure should
    learn residuals on top of the stronger GBM or state prior instead of
    fitting the full price path mostly from scratch.
- GBM residual refit after validation:
  - Config: `config/residual_calibrator_refit_q2.yaml`
  - Refit the residual calibrator through 2025-11 after validation-selected
    parameter search.
  - Test accuracy: `0.7646436863432745`
  - Result: refitting through validation degraded 2025-12 transfer and supports
    archiving GBM as a baseline rather than continuing small GBM postprocessors.
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
- Online previous-day spike-shape diagnostic:
  - Tested a leakage-safe rolling rule that lifts slots when the previous day
    had high prices in the same or neighboring slots.
  - Test-only best case improved the current online result only from
    `0.7857284442214671` to about `0.7868465739374848`.
  - Result: high-price persistence exists but is far too weak to close the 85%
    gap without stronger scarcity/fundamental signals.
- High-price classifier ranking diagnostic:
  - Trained LightGBM classifiers on 2025-01 through 2025-09 features to predict
    `Price >= 500` and `Price >= 800`, using 2025-10 through 2025-11 for
    threshold selection and 2025-12 for test.
  - The `Price >= 500` classifier had good 2025-12 ranking metrics
    (`AUC ~= 0.879`, `AP ~= 0.488`), but validation-selected thresholds did not
    transfer because December probabilities were much lower than validation
    probabilities.
  - Even with test-only Top-K/target selection, the classifier route topped out
    around `0.7751`, still below the online residual best `0.7857`.
  - Result: existing 2025 features contain some high-price ordering signal, but
    not enough magnitude/gating information to reach 85%.
- Dynamic online high-probability threshold diagnostic:
  - Tested recent-history quantile thresholds on `high_probability`,
    `cap_probability`, their maximum, and `y_pred`, lifting selected slots
    toward 400-600 yuan using only previous realized days.
  - On top of the base residual-calibrated model, validation-selected rules
    improved 2025-12 only from `0.7711041307390922` to about
    `0.7725372118177601`; test-only best within the representative grid reached
    about `0.7770261360458348`.
  - On top of the current online residual best, representative dynamic rules
    reached about `0.7877240483219137`, only `+0.0020` over the current best.
  - Result: recent-history threshold adaptation is a small possible
    postprocessor, but it is not a route to 85%.

## External Scarcity Signal Pipeline

Added `features.external_signals` and documented the expected schema in
`docs/external_scarcity_signals.md`. Forecast or disclosure columns that are
known before the day-ahead forecast can be merged directly. Realized variables
such as actual load, actual renewable output, actual generation, and realized
reserve margin are only exposed as previous-day lags and rolling statistics, so
the feature table remains leakage-safe.
`scripts/12_add_external_signals.py` can merge these columns into an existing
weather/previous-day-curve feature table once the external CSV is available.
`scripts/13_diagnose_high_price_regime.py` checks whether a feature table can
rank `Price >= 500` or `Price >= 800` slots well enough to justify a full
training run.

## Main Bottleneck

The December 2025 high-price and cap-price periods remain the bottleneck. Current
features can classify floor-price periods well, but high and cap probabilities
are not strong enough to lift predictions without damaging normal periods. The
best next improvement likely requires new leakage-safe external signals that
describe realized or forecast scarcity more directly, such as actual load,
renewable output, available capacity, reserve margin, outage/maintenance, or
official market disclosure data.

The current online residual best has `mean_relative_error=0.21427`; reaching
85% accuracy requires reducing this to `0.15`, a gap of about `0.06427`.
The largest remaining error share is not only the 800+ cap region: true
`200-500` yuan slots contribute about `42.2%` of total relative error, while
`500-800` and `800+` contribute about `12.0%` and `14.0%`. The worst hours are
morning and evening scarcity periods, especially hours 7-8, 0-1, and 17-22.
This supports focusing on scarcity-state features and mid/high-price magnitude,
not just cap-price classification.

`scripts/14_analyze_accuracy_gap.py` reproduces this gap analysis for any
candidate prediction file.

## External Data Availability Check

Public searches for Shaanxi 2025 spot-market 96-slot load, renewable output,
available capacity, reserve margin, outage, or market-scarcity disclosure data
did not find a directly downloadable machine-readable history file. The public
results mainly describe market operation, continuous settlement, rules, and
information-disclosure mechanisms. The codebase now includes
`scripts/15_audit_external_signal_file.py` to verify a supplied external CSV
before feature merging. In the current workspace, `data/external/scarcity_signals.csv`
is missing, so the external-signal route cannot yet be trained or evaluated.

## Validation-Selected Online Residual Tuning

Added `scripts/16_tune_online_residual_calibrator.py` to select online residual
parameters on 2025-10 through 2025-11 and evaluate the selected configuration
on fixed 2025-12. A 100-candidate run selected a conservative global residual
config with validation accuracy `0.818310971115351`, but its 2025-12 accuracy
was only `0.7758869665543694`, below the current hand-selected online best
`0.7857284442214671`. This confirms that the 2025-10/11 residual relationship
does not transfer strongly enough to solve the December gap.

## Daily Bias Correction Diagnostic

Added `scripts/17_diagnose_daily_bias_correction.py` to test whether daily
aggregate features can predict each day's residual bias. Using 2025-01 through
2025-09 for training and 2025-10 through 2025-11 for validation, the daily
residual-mean model had validation MAE near `6.05` but 2025-12 MAE near
`33.35`; applying the correction reduced 2025-12 accuracy from
`0.7711041307390922` to `0.6572` at shrink `0.25` and lower at larger shrink.
The residual-median variant was less damaging but still reduced accuracy to
about `0.7364` at shrink `0.25`. This rules out daily bias correction from the
current internal features as a viable path to 85%.

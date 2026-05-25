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

## Model

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

The quantile features are leakage-safe proxies because the current raw dataset
contains fundamental point forecasts but not the realized load/renewable series
needed to postprocess forecast errors. If realized fundamentals become available,
this module can be upgraded to the QR/HS probabilistic-input route described in
the literature.

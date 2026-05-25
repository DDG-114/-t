# External Scarcity Signal Features

The current internal dataset is not enough to identify December 2025 high-price
scarcity states reliably. The best verified internal-data model reaches
`accuracy=0.7857`, but it predicts only `14` slots above `500` when the test
month contains `426` true slots above `500`.

Electricity price forecasting literature usually treats spikes as a separate
regime driven by market fundamentals: load, renewable output, available
capacity, reserve margin, outages, interconnection constraints, and market
scarcity disclosures. The project now supports these signals through
`features.external_signals`.

## Input Schema

Place the external file outside git-tracked data, for example:

```text
data/external/scarcity_signals.csv
```

The file can use either:

- `Date`: full timestamp for each 15-minute slot; or
- `date` plus `slot`: delivery date and 0-95 slot index.

Recommended columns:

```text
reserve_margin_forecast
available_capacity_forecast
load_forecast_external
renewable_forecast_external
outage_capacity_declared
market_scarcity_index
actual_load
actual_renewable
actual_generation
actual_tie_line
realized_reserve_margin
```

## Leakage Rules

`direct_columns` are used at the target timestamp. Only put values here if they
are known before the day-ahead forecast is made, such as published reserve
margin forecasts or declared outage capacity.

`lagged_actual_columns` are never used at the current timestamp. The feature
builder only exposes same-slot lags and rolling statistics from previous days,
then drops the raw current actual value. This is the correct setting for
realized load, realized renewable output, actual generation, and realized
reserve margin.

Use `config/external_scarcity_template.yaml` as the starting config once real
scarcity/fundamental data is available.

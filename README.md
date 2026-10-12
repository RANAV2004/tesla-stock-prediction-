# Tesla Stock Analytics: Risk, Return, Earnings & Forecasting

An end-to-end data analysis of Tesla (TSLA) since 2015: raw data cleaning, SQL analysis, statistical
testing, a Power BI report, an interactive Streamlit dashboard, and an honestly evaluated LSTM forecast.

**Live dashboard:** *add your Streamlit link here after deploying (see "Deploy" below)*
**Power BI report:** [`powerbi/`](powerbi), with the PDF export and screenshots

**Stack:** Python (pandas, SciPy) · SQL (SQLite, window functions) · Power BI (DAX, star schema) ·
Streamlit + Plotly · TensorFlow/Keras · Jupyter

![Growth of $10,000](reports/figures/01_growth_of_10k.png)

## What makes this project different

Most published Tesla LSTM studies train a model on price levels, report a low RMSE, and stop there.
This project **benchmarks every model against a random walk** ("tomorrow = today") and tests whether
the difference is statistically real. Result: the price-level LSTM looks accurate on a chart but has a
**2.35× higher error than the naive forecast** (RMSE $22.50 vs $9.56, Diebold–Mariano p < 0.001). See
the [research paper](research/) for the full study.

## Key findings

<!-- FINDINGS:START -->
*Run `python run_pipeline.py` to fill this section with numbers from the latest data.*
<!-- FINDINGS:END -->

## Workflow

```
Yahoo Finance ──► data/raw/        untouched downloads (messy headers, text numbers, partial days)
                    │  src/clean_data.py  (logged: reports/cleaning_log.csv)
                    ▼
                 data/clean/       typed, validated, de-duplicated, anomalies flagged
                    │  src/build_db.py
                    ▼
                 SQLite  ──► 12 SQL analyses (sql/analysis/)
                    │
       ┌────────────┼──────────────┬───────────────────┐
       ▼            ▼              ▼                   ▼
  Notebooks    Streamlit app   Power BI report    LSTM forecast
  (analysis)   (app.py)        (powerbi/)         (src/forecast.py)
```

## 1. Data cleaning

Raw files are saved exactly as Yahoo Finance returns them. `src/clean_data.py` fixes them, and
[`notebooks/01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb) walks through each step with before/after views.

| Step | Problem handled | Decision |
|---|---|---|
| Structure | yfinance's 3-row header (`Price` / `Ticker` / `Date` lines) | Parse the real header, drop metadata rows |
| Types | Numbers stored as text, values like `null`, timezone-stamped earnings dates | Convert; invalid → missing; timestamps to New York time |
| Duplicates | Repeated trading dates / earnings rows | Keep one |
| Missing values | Empty rows, missing close, missing adj. close / OHLC / volume | Remove rows without a close (**prices are never interpolated**); fill the rest with stated rules |
| Validity | Non-positive prices, high below open/close, future dates | Remove or repair (high/low rebuilt from OHLC) |
| Completeness | Partial intraday row for today, stale repeated prices with no volume | Remove |
| Anomalies | Possible unadjusted splits (>40% day), moves beyond 5σ | **Flag, don't delete**: extreme moves are real market events |
| Earnings | Unreported future releases, wrong surprise %, pre-2015 rows | Remove / recompute |
| Alignment | Trading days missing for some tickers | Logged; analysis compares only shared days |

Every action and the number of rows it touched is in `reports/cleaning_log.csv`. The same validation
checks run before and after cleaning (`reports/data_quality.csv`); the "after" counts must all be zero.

## 2. Questions answered

| Area | Question | Where |
|---|---|---|
| Performance | How did $10,000 in Tesla compare with the S&P 500, Nasdaq, Ford and GM? | Analysis §2, SQL 01-02 |
| Risk | How volatile is Tesla, and how deep and long were its drawdowns? | Analysis §4, SQL 05-06 |
| Behaviour | How often do extreme days happen? Are returns normally distributed? | Analysis §5, SQL 03 |
| Market link | How much does Tesla move when the market moves (beta)? Is that changing? | Analysis §6, SQL 12 |
| Seasonality | Are some weekdays or months reliably better? Is the effect statistically real? | Analysis §7, SQL 04, 07 |
| Activity | When did volume spike? Longest winning/losing streaks? | SQL 08-09 |
| Earnings | How big are earnings-day moves? Does beating EPS estimates predict the reaction? | Analysis §8, SQL 10-11 |
| Forecasting | Can an LSTM beat "tomorrow = today"? | Analysis §9, `src/forecast.py` |

Full analysis: [`notebooks/02_tesla_analysis.ipynb`](notebooks/02_tesla_analysis.ipynb)

## 3. SQL

12 queries in [`sql/analysis/`](sql/analysis), each headed by the business question it answers.
Techniques: CTEs, `LAG`, `ROW_NUMBER`, `RANK`, running `MAX() OVER`, moving averages with `ROWS BETWEEN`
frames, gaps-and-islands for streaks, conditional aggregation to pivot, correlated subqueries, and
correlation/beta computed from aggregate moments.

## 4. Power BI report

Star schema (`DimDate`, `DimTicker` + 7 fact tables) exported by `src/export_powerbi.py`, ~30 DAX
measures (CAGR, volatility, Sharpe, beta, growth of $10K, event-study CAAR, forecast RMSE), and a 5-page
report: Overview, Risk & Returns, Earnings, Forecast, Data Quality.
See [`powerbi/BUILD_GUIDE.md`](powerbi/BUILD_GUIDE.md).

<!-- After building: add powerbi screenshots here, e.g.
![Power BI overview](reports/figures/powerbi_1_overview.png) -->

## 5. Streamlit dashboard

`streamlit run app.py`: six tabs, filtered by a date-range sidebar.
Overview (KPIs, growth of $10K, candlestick + moving averages) · Risk & returns (drawdowns, rolling
volatility and beta, return distribution, monthly heatmap) · Earnings (event study, every release) ·
SQL insights (run any of the 12 queries, see the SQL) · Forecast · Data cleaning (log + before/after validation).

## 6. Forecasting, done honestly

- Data split **by date**: train → validation (early stopping) → most recent 20% held out for testing.
- Scaler fitted on **training data only**, so nothing from the test period leaks in.
- The LSTM reads 60 days of **log returns** and predicts the next day's return, converted back to a price.
- Compared with the **naive forecast** (tomorrow = today) and, for direction, with **always guessing "up"**.

*Development note:* the first version predicted raw price levels and was about 2.6× worse than the
naive forecast on a single 80/20 test split (RMSE $22.50 vs $8.64), because test-period prices moved outside
the range seen in training. Switching to returns fixed that.

## 7. Research paper

[`research/`](research): an IEEE-format paper, *LSTM vs. Random Walk for Tesla Stock Forecasting*, with
fully reproducible experiments (`python research/run_experiments.py`).

- **Models:** random walk, drift, ARIMA (AIC-selected), Ridge on 20 lags, LSTM on price levels, LSTM on returns.
- **Evaluation:** leakage-free rolling-origin tests on 2022, 2023 and 2024, five random seeds each.
- **Metrics and tests:** RMSE, MAE, Theil's U, Diebold–Mariano (HLN correction), Pesaran–Timmermann.

| Model | Pooled RMSE | Theil's U | Verdict |
|---|---|---|---|
| Random walk | $9.56 | 1.00 | Baseline |
| LSTM (price levels) | $22.50 | 2.35 | Significantly worse (DM p < 0.001) |
| LSTM (returns) | ≈ $9.5 | ≈ 1.00 | No better than the random walk |
| ARIMA / Ridge / Drift | ≈ $9.5 | 0.996–1.007 | No better than the random walk |

Direction: the returns LSTM called the next day's direction correctly 52.8% of the time, vs 51.3% for
always guessing "up" (Pesaran–Timmermann p = 0.079, not significant). Conclusion: on daily price history
alone, Tesla behaves like a random walk, and published "accurate" LSTM results usually come from
missing baselines.

## Methods

- Returns use adjusted close. CAGR, annualized volatility (×√252), Sharpe/Sortino with a flat 3%
  risk-free rate (an assumption, in `config.py`).
- Statistical tests: Jarque-Bera (normality), Kruskal-Wallis (calendar effects), Mann-Whitney U
  (earnings-day move size), one-sample t-test (average earnings reaction). Non-parametric tests are used
  because returns are clearly non-normal.
- Event study: market-adjusted abnormal return (Tesla − S&P 500), −5 to +5 trading days.
- Earnings timing: Tesla reports after the 4pm close, so each release maps to the next session.

## Project structure

```
├── config.py               # tickers, dates, paths, thresholds (one place to change)
├── run_pipeline.py         # runs everything end to end
├── app.py                  # Streamlit dashboard
├── src/
│   ├── download_data.py    # Yahoo Finance -> data/raw/ (untouched)
│   ├── clean_data.py       # data/raw/ -> data/clean/ + cleaning log
│   ├── build_db.py         # data/clean/ -> SQLite
│   ├── run_sql.py          # sql/analysis/*.sql -> reports/sql/
│   ├── analysis.py         # returns, risk, drawdowns, beta, event study, tests
│   ├── forecast.py         # LSTM + evaluation vs naive baseline
│   ├── export_powerbi.py   # star schema -> powerbi/data/
│   └── db.py, style.py
├── sql/                    # schema.sql + analysis/ (12 queries)
├── notebooks/              # 01_data_cleaning, 02_tesla_analysis
├── powerbi/                # data/, measures.dax, tesla_theme.json, BUILD_GUIDE.md
├── data/raw/, data/clean/
├── research/               # IEEE paper, experiments, figures
├── models/                 # trained LSTM + scaler
└── reports/                # figures, SQL outputs, cleaning log, metrics, key_findings.md
```

## Run it locally

Python 3.12 recommended (TensorFlow doesn't support the very newest Python versions yet).

```bash
git clone https://github.com/RANAV2004/tesla-stock-prediction-.git
cd tesla-stock-prediction-
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

python run_pipeline.py      # download, clean, DB, SQL, model, notebooks, Power BI export (~5-10 min)
streamlit run app.py
```

Single steps: `python src/download_data.py`, `src/clean_data.py`, `src/build_db.py`, `src/run_sql.py`,
`src/forecast.py`, `src/export_powerbi.py`. Rerun without downloading: `python run_pipeline.py --skip-download`.

## Deploy (Streamlit Community Cloud, free)

1. Run the pipeline locally and push everything to GitHub, including `data/raw/`, `data/clean/`,
   `models/` and `reports/` (the app reads these; `data/tesla.db` is rebuilt automatically).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, **Create app**.
3. Repository `RANAV2004/tesla-stock-prediction-`, branch `main`, main file `app.py`.
   Under **Advanced settings** choose Python **3.12**. Pick a custom URL such as `tesla-analytics`.
4. **Deploy.** The first build takes a few minutes (TensorFlow is large). Then add the link at the top
   of this README, on your portfolio site, LinkedIn and resume.

To refresh the data later: rerun the pipeline locally and push; the app redeploys on its own.

## Limitations

- Yahoo Finance is a free, unofficial source; earnings dates cover recent years only.
- Flat risk-free rate; market-adjusted (not market-model) abnormal returns.
- The forecast uses price history only. It is a demonstration, **not investment advice**.

---
Built by **Vansh Rana** · [LinkedIn](https://linkedin.com/in/vanshrana-ai) · [GitHub](https://github.com/RANAV2004)

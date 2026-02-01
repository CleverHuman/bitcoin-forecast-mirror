# Code Review: forecast_signals.py

Assessment of **best practices**, **directory structure**, **modular design**, and **testability**.

---

## What’s Working Well

### 1. **Good use of `src/` package**

- **`src/db`**: Databricks connector isolated and reusable.
- **`src/metrics`**: Halving/decay logic in `halving.py` and `decay.py` with clear `__all__`.
- **`src/models`**: Ensemble, signals, backtest, cycle features in separate modules with a single public API via `src.models.__init__`.

Orchestration stays in `forecast_signals.py`; core logic lives in `src`. That’s a solid modular split.

### 2. **Thin orchestration**

`main()` is a clear pipeline: fetch → cycle metrics → train → signals → current signal → backtest → summary → save → plot. Real logic is in `src`; the script mostly wires and configures.

### 3. **Single entry point**

One `main()` and one `if __name__ == "__main__"` keep the script easy to run and reason about.

### 4. **Config from environment**

`FORECAST_DAYS` from env (with default) and `REPORTS_DIR` from `Path(__file__).parent` are reasonable and easy to change.

### 5. **Report capture**

`TeeOutput` for capturing stdout into the report file while still printing is a good fit for a “run and save report” workflow.

---

## Issues and Recommendations

### 1. **Duplicate `fetch_btc_data` (high impact)**

**Issue:** The same `fetch_btc_data()` is copy-pasted in:

- `forecast_signals.py`
- `forecast.py`
- `forecast_cycle.py`

**Recommendation:** Move it into the project once, e.g.:

- **Option A:** `src/db/btc_data.py` (or `src/data/btc.py`) with `fetch_btc_data(start_date, end_date)` that uses `DatabricksConnector` and returns a DataFrame with `ds`, `y`.
- **Option B:** Add a method `DatabricksConnector.fetch_btc_daily(start_date, end_date)` in `connector.py` if you want to keep all DB access in `db/`.

Then have all three scripts import and call that single implementation. This improves consistency, testability, and future changes (e.g. table or column renames).

---

### 2. **`forecast_signals.py` is doing too many jobs (modularity)**

**Issue:** The script mixes:

- Data fetching (should live in `src`)
- Plotting (~250 lines in `plot_signals`)
- Report formatting (`print_signal_summary`)
- Utility (`TeeOutput`)
- Orchestration (`main()`)

**Recommendation:**

- **Data:** Move `fetch_btc_data` to `src` as above.
- **Plotting:** Extract `plot_signals()` (and helpers) into e.g. `src/viz/signals_plot.py` or `scripts/plot_signals.py`. That makes it easier to:
  - Reuse the same plots from other scripts or a CLI.
  - Test or stub “no plot” in automated runs.
  - Change plot style in one place.
- **Reporting:** Either keep `print_signal_summary` in the script or move to e.g. `src/reporting.py` / `src/metrics/reporting.py` if you add more report formats later.
- **TeeOutput:** Move to e.g. `src/utils/tee.py` (or a small `src/utils.py`) so it can be reused and tested.

After this, `forecast_signals.py` becomes a thin orchestrator that imports from `src` and stays easy to read and change.

---

### 3. **Lazy import removed**

**Issue:** `plot_signals` used a lazy `from src.metrics.halving import backtest_predictions` inside the function, while the rest of the file uses top-level imports from `src.metrics`.

**Change made:** Use top-level `from src.metrics import backtest_predictions` and remove the import from inside `plot_signals`. This keeps imports consistent and avoids circular import surprises.

---

### 4. **No tests (testability)**

**Issue:** There are no tests (no `tests/` or `*_test.py` / `test_*.py`).

**Recommendation:**

- Add a `tests/` directory at repo root.
- Start with **unit tests** for pure, stateless functions that already live in `src`:
  - `add_cycle_features`, `get_cycle_phase` (e.g. `src/models/cycle_features.py`)
  - `generate_signals` (with a small fixture DataFrame)
  - `compute_cycle_metrics`, `compute_halving_averages` (with minimal date/price DataFrames)
- Use **pytest** and, if you want, **pandas.testing** for DataFrame outputs.
- Optionally add a **small integration test** that runs `main()` with a **mocked or fixture data source** (no real Databricks) and checks that outputs exist and have expected columns; that would protect the pipeline as you refactor.

Once `fetch_btc_data` is in `src`, you can also test the pipeline by injecting a fake fetcher that returns a fixed DataFrame.

---

### 5. **SQL and data fetching (best practice)**

**Issue:** `fetch_btc_data` builds SQL with f-strings:

```python
AND to_date(date) > '{start_date}'
AND to_date(date) < '{end_date}'
```

If `start_date`/`end_date` ever came from user input, this would be a risk. Right now they are from code/env, so the risk is low.

**Recommendation:** Keep dates from code or env. If you later allow user-supplied dates, validate/parse them (e.g. to `datetime`) and consider parameterized queries if the Databricks client supports them.

---

### 6. **Plot logic duplication**

**Issue:** In `plot_signals`, halving lines and tops/bottoms are drawn in both the 3-panel figure and the linear-scale figure with almost the same code.

**Recommendation:** Extract small helpers, e.g.:

- `_draw_halving_lines(ax, df_ds_min, forecast_ds_max)`
- `_draw_tops_bottoms(ax, cycle_metrics, last_historical)`

and call them from both figure-building blocks. That will shorten `plot_signals`, make it easier to change behavior in one place, and simplify any future tests or headless plotting.

---

### 7. **Dependency injection for data (optional, for tests)**

**Issue:** `fetch_btc_data` instantiates `DatabricksConnector()` inside the function, so the script is tied to real Databricks.

**Recommendation:** For easier testing and flexibility:

- Introduce a small abstraction, e.g. “data provider”: a function or callable `(start_date, end_date) -> pd.DataFrame`.
- Default implementation uses `DatabricksConnector` (or your new `src` fetch function).
- `main()` (or a runner) can accept an optional provider; tests pass a function that returns fixture DataFrames.

You can do this after centralizing `fetch_btc_data` in `src` and adding the first tests.

---

### 8. **Directory layout (structure)**

**Current:**

- Root: `forecast_signals.py`, `forecast.py`, `forecast_cycle.py`, `btc_prediction.py`, `src/`, `reports/`, `docs/`, `.notes/`.

**Assessment:** For this size project, multiple entry scripts at root is fine and common. No structural change is required.

**Optional:** If the number of scripts grows, you could move them under `scripts/` (e.g. `scripts/forecast_signals.py`) and run them as `python -m scripts.forecast_signals` or via a small CLI in `scripts/`. Not necessary until you feel root is crowded.

---

## Summary

| Area              | Verdict | Notes                                                                 |
|-------------------|--------|-----------------------------------------------------------------------|
| Best practices    | Good   | Env config, single main, clear flow. Improve: centralize data fetch, avoid SQL string interpolation if inputs ever become user-supplied. |
| Directory structure | Good | `src/` with db, metrics, models is clear. Scripts at root are fine.  |
| Modular design    | Good   | Logic in `src`, script orchestrates. Improve: move fetch + optional viz/report/tee into `src` to make script thinner. |
| Testability       | Weak   | No tests yet. Logic in `src` is easy to unit test; add `tests/` and pytest, then optional integration test with fake data. |
| Ease of change    | Good   | Changing weights, horizons, or model choices is straightforward. Duplicate fetch and fat script make some changes riskier; above refactors improve that. |

**Suggested order of work**

1. Move `fetch_btc_data` into `src` (e.g. `src/db/btc_data.py` or `DatabricksConnector.fetch_btc_daily`) and use it from all three scripts.
2. Add `tests/` and a few unit tests for `src` (cycle features, signals, metrics).
3. Optionally extract `plot_signals` (and helpers) and `TeeOutput` into `src` to make `forecast_signals.py` a thin, easy-to-change orchestrator.

These steps keep your current structure and design while making the codebase more consistent, testable, and easier to change over time.

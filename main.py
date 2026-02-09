import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.db import fetch_btc_data
from src.metrics import compute_cycle_metrics, compute_halving_averages, print_halving_summary
from src.models import backtest_signals, generate_signals, get_current_signal, train_simple_ensemble
from src.reporting import print_signal_summary

load_dotenv()

# Forecast config
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "365"))
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
BACKTEST_POSITION_SIZE = float(os.getenv("BACKTEST_POSITION_SIZE", "1.0"))

# Reports directory
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Create a FastAPI instance
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:5173", "http://127.0.0.1:8000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    from_date: Optional[str] = None
    days: Optional[int] = None
    no_signals: bool = False


def make_json_serializable(obj):
    """Convert any object to JSON-serializable format."""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'isoformat'):
        # Handle pandas Timestamp or datetime
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    else:
        # Convert any other object to string
        return str(obj)


@app.get("/")
def read_root():
    """
    Simple health-check endpoint.
    """
    return {"message": "OK"}


@app.get("/api/example")
def read_example():
    """
    Example API endpoint used by the frontend.
    """
    return {
        "id": 1,
        "name": "Example from FastAPI",
        "description": "This data comes from the FastAPI backend at /api/example.",
    }


async def generate_forecast_sse(request: ForecastRequest):
    """
    Generate forecast with SSE progress updates.
    """
    forecast_days = request.days or FORECAST_DAYS
    from_date = request.from_date
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def send_event(event_type: str, data: dict):
        """Helper to format SSE message"""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    try:
        # Step 1: Fetch data
        yield send_event("progress", {"step": 1, "total": 6, "message": "Fetching BTC data..."})
        await asyncio.sleep(0.1)  # Allow event to be sent

        start_date = "2015-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        df_full = fetch_btc_data(start_date, end_date)
        df_full = df_full.sort_values("ds").reset_index(drop=True)

        # Split data if backtesting
        if from_date:
            forecast_start = pd.to_datetime(from_date)
            df_train = df_full[df_full["ds"] < forecast_start].copy()
            df_actual = df_full[df_full["ds"] >= forecast_start].copy()

            if df_train.empty:
                yield send_event("error", {"message": f"No data before {from_date}"})
                return

            df = df_train
        else:
            df = df_full
            df_actual = None

        # Step 2: Compute cycle metrics
        yield send_event("progress", {"step": 2, "total": 6, "message": "Computing cycle metrics..."})
        await asyncio.sleep(0.1)

        cycle_metrics = compute_cycle_metrics(df)
        averages = compute_halving_averages(cycle_metrics=cycle_metrics)

        # Step 3: Train model
        yield send_event("progress", {"step": 3, "total": 6, "message": f"Training model (forecasting {forecast_days} days)..."})
        await asyncio.sleep(0.1)

        forecast = train_simple_ensemble(
            df,
            periods=forecast_days,
            halving_averages=averages,
            cycle_metrics=cycle_metrics,
        )

        # Step 4: Generate signals
        if not request.no_signals:
            yield send_event("progress", {"step": 4, "total": 6, "message": "Generating buy/sell signals..."})
            await asyncio.sleep(0.1)

            df_signals = generate_signals(df, cycle_weight=0.4, technical_weight=0.6)
            current = get_current_signal(df_signals, cycle_metrics=cycle_metrics, averages=averages)

            yield send_event("progress", {"step": 5, "total": 6, "message": "Running backtest..."})
            await asyncio.sleep(0.1)

            backtest = backtest_signals(df_signals, initial_capital=INITIAL_CAPITAL, position_size=BACKTEST_POSITION_SIZE)
        else:
            df_signals = df.copy()
            current = None
            backtest = None

        # Step 6: Save results
        yield send_event("progress", {"step": 6, "total": 6, "message": "Saving results..."})
        await asyncio.sleep(0.1)

        suffix = f"_{from_date}" if from_date else ""
        signals_path = REPORTS_DIR / f"signals_{timestamp}{suffix}.csv"
        forecast_path = REPORTS_DIR / f"forecast_{timestamp}{suffix}.csv"

        df_signals.to_csv(signals_path, index=False)
        forecast.to_csv(forecast_path, index=False)

        # Prepare final response
        response = {
            "timestamp": timestamp,
            "forecast_days": forecast_days,
            "from_date": from_date,
            "data_points": len(df),
            "forecast_file": forecast_path.name,
            "signals_file": signals_path.name,
        }

        # Add optional fields with JSON serialization
        if current:
            response["current_signal"] = make_json_serializable(current)

        if backtest:
            response["backtest"] = make_json_serializable(backtest)

        # Send completion event
        yield send_event("complete", response)

    except Exception as e:
        yield send_event("error", {"message": str(e)})


@app.post("/api/forecast/run")
async def run_forecast(request: ForecastRequest):
    """
    Run BTC forecast with cycle-aware signals using SSE for progress updates.

    Args:
        request: ForecastRequest with optional from_date, days, no_signals

    Returns:
        SSE stream with progress updates
    """
    return StreamingResponse(
        generate_forecast_sse(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.get("/api/forecast/history")
def get_forecast_history():
    """
    Get history of all forecast runs.

    Returns:
        List of forecast files with metadata
    """
    try:
        forecast_files = sorted(REPORTS_DIR.glob("forecast_*.csv"), reverse=True)

        history = []
        for file_path in forecast_files:
            # Parse filename: forecast_YYYYMMDD_HHMMSS[_from-date].csv
            name_parts = file_path.stem.split("_")

            if len(name_parts) >= 3:
                timestamp_str = f"{name_parts[1]}_{name_parts[2]}"
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                    # Check if there's a from_date suffix
                    from_date = name_parts[3] if len(name_parts) > 3 else None

                    # Read first few rows to get info
                    df = pd.read_csv(file_path, nrows=5)

                    history.append({
                        "filename": file_path.name,
                        "timestamp": timestamp.isoformat(),
                        "from_date": from_date,
                        "forecast_points": len(pd.read_csv(file_path)),
                        "file_size": file_path.stat().st_size,
                    })
                except Exception:
                    continue

        return {
            "total": len(history),
            "forecasts": history
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/forecast/file/{filename}")
def get_forecast_file(filename: str):
    """
    Serve a forecast CSV file.

    Args:
        filename: The name of the forecast file (e.g., forecast_20260209_002000.csv)

    Returns:
        The CSV file content
    """
    try:
        from fastapi.responses import FileResponse

        # Security check: only allow files in reports directory and prevent path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return {"error": "Invalid filename"}

        file_path = REPORTS_DIR / filename

        if not file_path.exists():
            return {"error": f"File not found: {filename}"}

        if not file_path.is_file():
            return {"error": "Not a file"}

        return FileResponse(
            path=str(file_path),
            media_type="text/csv",
            filename=filename
        )

    except Exception as e:
        return {"error": str(e)}
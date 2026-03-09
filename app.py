from flask import Flask, render_template, request
import pandas as pd
from data_processor import process_files
from predictor import run_forecast

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/forecast", methods=["POST"])
def forecast():

    try:

        # ── Validate files ───────────────────────────────────────
        files = request.files.getlist("files")
        files = [f for f in files if f and f.filename.strip() != ""]
        print("\nFILES RECEIVED:", [f.filename for f in files])
        if not files:
            return render_template(
                "index.html",
                error="No files uploaded. Please select at least one file."
            ), 400

        # ── Form values ──────────────────────────────────────────
        target = request.form.get("target", "Net Cash Flow")
        months = int(request.form.get("months", 12))

        # ── Process files ────────────────────────────────────────
        print("STEP 1 — files received")

        df = process_files(files)
        if len(df) < 3:
            raise ValueError(
                "Not enough historical data. Please upload at least "
                "3-5 cash flow statements (multiple years)."
            )
        print("STEP 2 — dataframe created\n")

        print("DATAFRAME AFTER EXTRACTION:")
        print(df)

        print("ROW COUNT:", len(df))
        # ── Run forecast ─────────────────────────────────────────
        forecast_df = run_forecast(df, target, months)

        print("STEP 3 — forecast completed")
        # ── Build historical series ──────────────────────────────
        df["Date"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Month"].astype(str),
            errors="coerce"
        )
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")

        historical_dates  = df["Date"].dt.strftime("%Y-%m").tolist()
        historical_values = df[target].tolist()

        # ── Build forecast series ────────────────────────────────
        forecast_dates  = forecast_df["Date"].dt.strftime("%Y-%m").tolist()
        forecast_values = forecast_df[f"Predicted {target}"].tolist()

        # ── Combine for Chart.js (pad with None) ─────────────────
        combined_labels   = historical_dates + forecast_dates
        historical_series = historical_values + [None] * len(forecast_dates)
        forecast_series   = [None] * len(historical_dates) + forecast_values

        return render_template(
            "result.html",
            target=target,
            labels=combined_labels,
            historical_series=historical_series,
            forecast_series=forecast_series,
            forecast_table=forecast_df.to_dict(orient="records")
        )

    except ValueError as e:
        print("PIPELINE ERROR:", e)
        return str(e), 400

    except Exception as e:
        print("SERVER ERROR:", e)
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
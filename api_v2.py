# api_v2.py — FastAPI backend (multi-method + MAPE + ensemble) with CORS for demandplanningcourse.com
import io, base64
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from arima_core_v2 import read_input_df, run_all_methods_for_group, REQUIRED_COLS

app = FastAPI(title="ARIMA/SARIMA API v2 — multi-method + MAPE + ensemble")

# CORS: allow your site origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://demandplanningcourse.com",
        "https://www.demandplanningcourse.com",
        # add Tilda preview domain if needed, e.g. "https://*.tilda.ws"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...), sheet: Optional[str] = Form(None)):
    content = await file.read()
    if file.filename.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet) if sheet else pd.read_excel(io.BytesIO(content))
    elif file.filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    else:
        return {"ok": False, "error": "Unsupported file type"}

    try:
        df = read_input_df(df)
    except Exception as e:
        return {"ok": False, "error": f"Bad columns: {e}"}

    accounts = sorted(df["Account"].dropna().unique().tolist())
    products_by_acc = {acc: sorted(df.loc[df["Account"]==acc, "Product"].dropna().unique().tolist()) for acc in accounts}
    return {"ok": True, "accounts": accounts, "products_by_account": products_by_acc}

@app.post("/forecast")
async def forecast(file: UploadFile = File(...),
                   account: str = Form(...),
                   product: str = Form(...),
                   horizon: int = Form(36),
                   sheet: Optional[str] = Form(None)):
    content = await file.read()
    if file.filename.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet) if sheet else pd.read_excel(io.BytesIO(content))
    elif file.filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    else:
        return {"ok": False, "error": "Unsupported file type"}

    df = read_input_df(df)
    g = df[(df["Account"]==account) & (df["Product"]==product)].copy().sort_values("Date")
    if g.empty or g["Actual"].notna().sum() < 6:
        return {"ok": False, "error": "Not enough points (need ≥ 6)"}

    res = run_all_methods_for_group(g, horizon=int(horizon))

    # Build Excel (all outputs)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        g.to_excel(writer, index=False, sheet_name="Original")
        res["cleaned_df"].to_excel(writer, index=False, sheet_name="Cleaned")
        res["final_df"].to_excel(writer, index=False, sheet_name="Forecast_Final")
        for m, dfm in res["methods"].items():
            dfm.to_excel(writer, index=False, sheet_name=f"Forecast_{m[:28]}")
        res["summary"].to_excel(writer, index=False, sheet_name="Methods_Summary")
        analysis_df = pd.DataFrame([{
            "Account": account,
            "Product": product,
            "Chosen": res["analysis"]["chosen"],
            "Best_Single": res["analysis"]["best_single"],
            "Strong_Seasonality": res["analysis"]["profile"]["strong_seasonality"],
            "Lag12_corr": res["analysis"]["profile"]["seasonal_lag12_corr"],
            "Len": res["analysis"]["profile"]["length"],
            "Zeros": res["analysis"]["cleaning"]["zeros"],
            "Zeros_to_NaN": res["analysis"]["cleaning"]["zero_to_nan"],
            "Clipped_upper": res["analysis"]["cleaning"]["clipped_upper"],
            "Clipped_lower": res["analysis"]["cleaning"]["clipped_lower"],
            "Ensemble_top": ", ".join(res["analysis"]["ensemble_top"] or []),
            "Ensemble_weights": ", ".join([f"{w:.3f}" for w in (res["analysis"]["ensemble_weights"] or [])])
        }])
        analysis_df.to_excel(writer, index=False, sheet_name="Analysis")

    buf.seek(0)
    excel_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # JSON payload for frontend
    history = [{"date": d.strftime("%Y-%m-%d"), "value": float(v) if pd.notna(v) else None}
               for d,v in zip(g["Date"], g["Actual"])]
    cleaned = [{"date": d.strftime("%Y-%m-%d"), "value": float(v) if pd.notna(v) else None}
               for d,v in zip(res["cleaned_df"]["Date"], res["cleaned_df"]["Actual_Clean"])]
    final = [{"date": d.strftime("%Y-%m-%d"), "value": float(v) if pd.notna(v) else None}
             for d,v in zip(res["final_df"]["Date"], res["final_df"]["Forecast"])]

    all_methods = {}
    for name, dfm in res["methods"].items():
        all_methods[name] = [{"date": d.strftime("%Y-%m-%d"), "value": float(v) if pd.notna(v) else None}
                             for d,v in zip(dfm["Date"], dfm["Forecast"])]

    summary = res["summary"].fillna("").to_dict(orient="records")

    return {
        "ok": True,
        "analysis": res["analysis"],
        "history": history,
        "cleaned": cleaned,
        "final": final,
        "methods": all_methods,
        "summary": summary,
        "excel_b64": excel_b64,
        "file_name": f"Forecast_{account}_{product}_all_methods.xlsx"
    }

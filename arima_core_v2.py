
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None

REQUIRED_COLS = ["Year", "Month number", "Account", "Product", "Actual"]

# ---------- Helpers ----------

def _safe_div(a, b):
    b = np.where(np.abs(b) < 1e-12, np.nan, b)
    return a / b

def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.array(actual, dtype=float)
    f = np.array(forecast, dtype=float)
    mask = (np.abs(a) > 1e-12) & np.isfinite(a) & np.isfinite(f)
    if mask.sum() == 0:
        return np.nan
    return float(np.nanmean(np.abs(_safe_div(a - f, a))) * 100.0)

def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.array(actual, dtype=float)
    f = np.array(forecast, dtype=float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    mask = (denom > 1e-12) & np.isfinite(a) & np.isfinite(f)
    if mask.sum() == 0:
        return np.nan
    return float(np.nanmean(np.abs(_safe_div(a - f, denom))) * 100.0)

def auto_corr_at_lag(y: pd.Series, lag: int = 12) -> float:
    yv = y.values.astype(float)
    if len(yv) <= lag + 1:
        return np.nan
    y1 = yv[:-lag]
    y2 = yv[lag:]
    if np.std(y1) < 1e-12 or np.std(y2) < 1e-12:
        return np.nan
    return float(np.corrcoef(y1, y2)[0,1])

# ---------- I/O ----------

def read_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["year", "год"]:
            colmap[c] = "Year"
        elif lc in ["month number", "month_number", "месяц", "месяц номер", "month"]:
            colmap[c] = "Month number"
        elif lc in ["account", "клиент", "customer"]:
            colmap[c] = "Account"
        elif lc in ["product", "sku", "продукт"]:
            colmap[c] = "Product"
        elif lc in ["actual", "sales", "qty", "кол-во", "объем"]:
            colmap[c] = "Actual"
    df = df.rename(columns=colmap)

    for r in REQUIRED_COLS:
        if r not in df.columns:
            raise ValueError(f"Missing required column: {r}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month number"] = pd.to_numeric(df["Month number"], errors="coerce").astype("Int64")
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
    df = df.dropna(subset=["Year","Month number","Account","Product"]).copy()
    df["Month number"] = df["Month number"].clip(lower=1, upper=12)
    df["Date"] = pd.to_datetime(dict(year=df["Year"].astype(int), month=df["Month number"].astype(int), day=1))
    return df

# ---------- Cleaning ----------

def clean_series(y: pd.Series) -> Tuple[pd.Series, Dict[str,int]]:
    """Return cleaned series and cleaning diagnostics (counts)."""
    s0 = y.sort_index().copy()
    diag = {"zeros": int((s0==0).sum()), "zero_to_nan": 0, "clipped_upper": 0, "clipped_lower": 0}

    med = np.nanmedian(s0.values)
    if med is not None and med > 0:
        zero_idx = s0.index[s0 == 0]
        if len(zero_idx) > 0 and (s0==0).mean() < 0.4:
            diag["zero_to_nan"] = int(len(zero_idx))
            s0.loc[zero_idx] = np.nan

    s = s0.interpolate(method="time").bfill().ffill()

    roll_med = s.rolling(window=3, center=True, min_periods=1).median()
    resid = s - roll_med
    mad = resid.abs().rolling(window=5, center=True, min_periods=1).median()
    upper = roll_med + 5 * mad
    lower = (roll_med - 5 * mad).clip(lower=0)

    before = s.copy()
    s = s.clip(lower=lower, upper=upper)
    diag["clipped_upper"] = int((before > upper).sum())
    diag["clipped_lower"] = int((before < lower).sum())

    s = s.clip(lower=0)
    return s, diag

# ---------- Methods ----------

def fc_naive(y: pd.Series, horizon: int) -> pd.Series:
    last = y.dropna().iloc[-1]
    idx = pd.date_range(start=y.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series([last]*horizon, index=idx)

def fc_snaive(y: pd.Series, horizon: int) -> pd.Series:
    idx = pd.date_range(start=y.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    if len(y) < 12:
        return fc_naive(y, horizon)
    out = []
    for d in idx:
        prev = y[y.index.month == d.month]
        if prev.empty:
            out.append(y.iloc[-1])
        else:
            out.append(prev.iloc[-1])
    return pd.Series(out, index=idx)

def fc_ma(y: pd.Series, horizon: int, window: int=12) -> pd.Series:
    m = y.dropna().tail(window).mean()
    idx = pd.date_range(start=y.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series([m]*horizon, index=idx)

def fc_ets(y: pd.Series, horizon: int, seasonal: bool=True, multiplicative: bool=False) -> Optional[pd.Series]:
    if ExponentialSmoothing is None:
        return None
    if seasonal and len(y) < 24:
        seasonal = False
    try:
        if seasonal:
            model = ExponentialSmoothing(
                y, trend='add', seasonal=('mul' if multiplicative else 'add'), seasonal_periods=12, initialization_method='estimated'
            ).fit(optimized=True, use_brute=True)
        else:
            model = ExponentialSmoothing(
                y, trend='add', seasonal=None, initialization_method='estimated'
            ).fit(optimized=True, use_brute=True)
        idx = pd.date_range(start=y.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        fc = model.forecast(horizon)
        fc.index = idx
        return fc
    except Exception:
        return None

def fit_best_sarima(y: pd.Series, seasonal: bool=True):
    if SARIMAX is None:
        return None, {"model":"Unavailable","order":None,"seasonal_order":None,"aic":np.nan}
    y = y.astype(float)
    best_aic = np.inf
    best_res = None
    best_cfg = None
    s = 12 if seasonal else 0

    p_values = [0,1,2]
    d_values = [0,1]
    q_values = [0,1,2]
    P_values = [0,1]
    D_values = [0,1]
    Q_values = [0,1]

    n = len(y.dropna())
    if n < 18:
        p_values = [0,1]
        q_values = [0,1]
        d_values = [0,1]
        P_values = [0]
        D_values = [0]
        Q_values = [0]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                if seasonal and s >= 2:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                try:
                                    res = SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,s),
                                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                                    if res.aic < best_aic:
                                        best_aic = res.aic
                                        best_res = res
                                        best_cfg = {"model":"SARIMA","order":(p,d,q),"seasonal_order":(P,D,Q,s),"aic":res.aic}
                                except Exception:
                                    continue
                else:
                    try:
                        res = SARIMAX(y, order=(p,d,q),
                                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_res = res
                            best_cfg = {"model":"ARIMA","order":(p,d,q),"seasonal_order":None,"aic":res.aic}
                    except Exception:
                        continue
    if best_cfg is None:
        return None, {"model":"None","order":None,"seasonal_order":None,"aic":np.nan}
    return best_res, best_cfg

def fc_sarima(y: pd.Series, horizon: int) -> Tuple[Optional[pd.Series], Dict]:
    seasonal_ok = len(y) >= 24
    model, cfg = fit_best_sarima(y, seasonal=seasonal_ok)
    if model is None:
        return None, cfg
    try:
        res = model.get_forecast(steps=horizon)
        fc = res.predicted_mean
        idx = pd.date_range(start=y.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        fc.index = idx
        return fc, cfg
    except Exception:
        return None, {"model":"MA_Fallback","order":None,"seasonal_order":None,"aic":np.nan}

# ---------- Profiling & Backtest ----------

def profile_series(y: pd.Series) -> Dict:
    n = int(y.notna().sum())
    seas12 = auto_corr_at_lag(y.fillna(0), 12)
    strong_season = (not np.isnan(seas12)) and (seas12 >= 0.3) and (n >= 24)
    return {"length": n, "seasonal_lag12_corr": seas12, "strong_seasonality": bool(strong_season)}

def static_holdout_mape(y: pd.Series, method_name: str, horizon: int=12, season: bool=True) -> Dict:
    """Train on first n-h, forecast H, compare to last H. Returns {'MAPE':..., 'sMAPE':..., 'H':H}"""
    n = len(y)
    H = min(horizon, max(1, n // 4))
    if H < 3:  # слишком мало для оценки
        return {"MAPE": np.nan, "sMAPE": np.nan, "H": H}
    train = y.iloc[:-H]
    test = y.iloc[-H:]
    fc = None

    if method_name == "Naive":
        fc = fc_naive(train, H)
    elif method_name == "SeasonalNaive":
        fc = fc_snaive(train, H)
    elif method_name == "MA6":
        fc = fc_ma(train, H, window=6)
    elif method_name == "MA12":
        fc = fc_ma(train, H, window=12)
    elif method_name == "ETS_add":
        fc = fc_ets(train, H, seasonal=season, multiplicative=False)
    elif method_name == "ETS_mul":
        fc = fc_ets(train, H, seasonal=season, multiplicative=True)
    elif method_name == "ARIMA_SARIMA":
        fc, _ = fc_sarima(train, H)

    if fc is None:
        return {"MAPE": np.nan, "sMAPE": np.nan, "H": H}

    # align
    fc = fc.iloc[:H]
    fc.index = test.index
    return {"MAPE": mape(test.values, fc.values), "sMAPE": smape(test.values, fc.values), "H": H}

# ---------- Main per-group pipeline ----------

def run_all_methods_for_group(g: pd.DataFrame, horizon: int=36) -> Dict:
    g = g.sort_values("Date")
    y = g.set_index("Date")["Actual"].astype(float)

    # Transform & clean
    use_log = (y > 0).all()
    yt = np.log1p(y) if use_log else y.copy()
    yt_clean, clean_diag = clean_series(yt)

    # Invert cleaning to original scale
    if use_log:
        y_clean = np.expm1(yt_clean).clip(lower=0)
    else:
        y_clean = yt_clean.clip(lower=0)

    # Profile
    prof = profile_series(y_clean)

    # Forecasts (history on cleaned series)
    methods = {}
    # naive & snaive
    methods["Naive"] = fc_naive(y_clean, horizon)
    methods["SeasonalNaive"] = fc_snaive(y_clean, horizon)
    # MA
    methods["MA6"] = fc_ma(y_clean, horizon, window=6)
    methods["MA12"] = fc_ma(y_clean, horizon, window=12)
    # ETS
    if ExponentialSmoothing is not None:
        methods["ETS_add"] = fc_ets(y_clean, horizon, seasonal=prof["strong_seasonality"], multiplicative=False)
        methods["ETS_mul"] = fc_ets(y_clean, horizon, seasonal=prof["strong_seasonality"], multiplicative=True)
    else:
        methods["ETS_add"] = None
        methods["ETS_mul"] = None
    # SARIMA
    fc_sari, sari_cfg = fc_sarima(y_clean, horizon)
    methods["ARIMA_SARIMA"] = fc_sari

    # Scores via static holdout
    scores = {}
    for m in methods.keys():
        scores[m] = static_holdout_mape(y_clean, m, horizon=12, season=prof["strong_seasonality"])

    # Pick best by MAPE (fallback to sMAPE if MAPE is nan)
    def metric_of(m):
        s = scores[m]
        if np.isnan(s["MAPE"]) and not np.isnan(s["sMAPE"]):
            return s["sMAPE"]
        return s["MAPE"]
    valid_methods = [m for m,v in methods.items() if v is not None]
    ranked = sorted(valid_methods, key=lambda m: (metric_of(m) if metric_of(m) is not None else np.inf))
    best = ranked[0] if ranked else "Naive"

    # Build Ensemble (top-3 by metric)
    top3 = ranked[:3]
    weights = []
    for m in top3:
        val = metric_of(m)
        if val is None or not np.isfinite(val) or val <= 0:
            weights.append(0.0)
        else:
            weights.append(1.0 / val)
    weights = np.array(weights, dtype=float)
    if weights.sum() <= 0:
        ensemble = methods.get(best)
        weights_norm = None
    else:
        weights_norm = weights / weights.sum()
        # aligned combine
        idx = methods[top3[0]].index
        mat = np.vstack([methods[m].reindex(idx).values for m in top3])
        ens_vals = np.dot(weights_norm, mat)
        ensemble = pd.Series(ens_vals, index=idx)

    # Decide final
    if len(top3) >= 2:
        gap = (metric_of(ranked[1]) - metric_of(ranked[0])) / max(1e-9, metric_of(ranked[0]))
        final_name = "Ensemble" if (weights_norm is not None and gap < 0.1) else best
    else:
        final_name = best

    final_fc = ensemble if final_name == "Ensemble" else methods[best]

    # Pack outputs
    cleaned_df = y_clean.reset_index().rename(columns={"Date":"Date", 0:"Actual_Clean"})
    cleaned_df["Actual_Clean"] = y_clean.values
    method_frames = {}
    for name, ser in methods.items():
        if ser is None:
            continue
        dfm = ser.reset_index()
        dfm.columns = ["Date","Forecast"]
        dfm["Year"] = dfm["Date"].dt.year
        dfm["Month number"] = dfm["Date"].dt.month
        method_frames[name] = dfm

    final_df = final_fc.reset_index()
    final_df.columns = ["Date","Forecast"]
    final_df["Year"] = final_df["Date"].dt.year
    final_df["Month number"] = final_df["Date"].dt.month

    # summary table
    summary_rows = []
    for m in methods.keys():
        sc = scores[m]
        summary_rows.append({
            "Method": m,
            "MAPE_holdout": sc["MAPE"],
            "sMAPE_holdout": sc["sMAPE"],
            "Holdout_H": sc["H"],
        })
    summary = pd.DataFrame(summary_rows).sort_values(["MAPE_holdout","sMAPE_holdout"], na_position="last")

    analysis = {
        "cleaning": clean_diag,
        "profile": prof,
        "chosen": final_name,
        "best_single": best,
        "sarima_cfg": sari_cfg if 'sari_cfg' in locals() else None,
        "ensemble_top": top3,
        "ensemble_weights": (weights_norm.tolist() if isinstance(weights_norm, np.ndarray) else None)
    }

    return {
        "cleaned_df": cleaned_df,
        "methods": method_frames,
        "final_df": final_df,
        "summary": summary,
        "analysis": analysis
    }

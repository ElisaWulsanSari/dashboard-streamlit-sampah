# model.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Prophet (opsional, kalau tidak ada akan otomatis di-skip)
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None  # type: ignore
    PROPHET_AVAILABLE = False
    pass

# =============================
# KONFIGURASI GLOBAL
# =============================
TARGET = "Estimasi_Timbulan_Plastik_Ton"
TIME_COL = "Tahun"
START_YEAR = 2019
TRAIN_END = 2022  # <= 2022: train, > 2022: test

# Nama file (HARUS sama dengan notebook)
FILE1 = "Data_Timbulan_Sampah_SIPSN_KLHK.xlsx"
FILE2 = "Data_Komposisi_Jenis_Sampah_SIPSN_KLHK (1).xlsx"
FILE3 = "Data_Capaian_SIPSN_KLHK.xlsx"
FILE4 = "Data TPS  TPS 3R.xlsx"
FILE5 = "Data Fasilitas TPATPST.xlsx"
FILE6 = "Data-dukcapil_jumlah_penduduk_kabkot-2025-11-26-1764141751067.xlsx"


# =============================
# 1. LOAD DATA MENTAH
# =============================
def load_raw_data(base_path="."):
    """Load keenam file Excel persis seperti di notebook."""
    base_path = Path(base_path)

    df1 = pd.read_excel(base_path / FILE1, sheet_name="Sheet1", header=1)
    df2 = pd.read_excel(base_path / FILE2, sheet_name="Sheet1", header=1)
    df3 = pd.read_excel(base_path / FILE3, sheet_name="Sheet1", header=1)
    df4 = pd.read_excel(base_path / FILE4, sheet_name="Sheet1", header=1)
    df5 = pd.read_excel(base_path / FILE5, sheet_name="Sheet1", header=1)
    df6 = pd.read_excel(base_path / FILE6, sheet_name="Sheet 1", header=2)

    return {
        "df1_timbulan": df1,
        "df2_komposisi": df2,
        "df3_capaian": df3,
        "df4_tps3r": df4,
        "df5_tpatpst": df5,
        "df6_penduduk": df6,
    }


# =============================
# 2. FUNGSI BANTUAN CLEANING
# =============================
def clean_numeric(series: pd.Series) -> pd.Series:
    """Hapus koma dan ubah ke float, NaN -> 0."""
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce").fillna(0)


def clean_location(series: pd.Series) -> pd.Series:
    """Standarisasi nama lokasi (uppercase + buang awalan Kab/Kota)."""
    s = series.astype(str).str.upper().str.strip()
    s = s.str.replace(r"^(KAB\.|KOTA\s|KAB\s|KOTA\.)\s*", "", regex=True)
    return s


# =============================
# 3. CLEANING PER FILE
# =============================
def preprocess_each_file(raw: dict):
    df1 = raw["df1_timbulan"].copy()
    df2 = raw["df2_komposisi"].copy()
    df3 = raw["df3_capaian"].copy()
    df4 = raw["df4_tps3r"].copy()
    df5 = raw["df5_tpatpst"].copy()
    df6 = raw["df6_penduduk"].copy()

    # --- FILE 1: Timbulan ---
    df1_sel = df1[["Tahun", "Provinsi", "Kabupaten/Kota", "Timbulan Sampah Tahunan(ton)"]].copy()
    df1_sel["Timbulan Sampah Tahunan(ton)"] = clean_numeric(df1_sel["Timbulan Sampah Tahunan(ton)"])
    df1_sel["Key_Prov"] = clean_location(df1_sel["Provinsi"])
    df1_sel["Key_Kab"] = clean_location(df1_sel["Kabupaten/Kota"])

    # --- FILE 2: Komposisi (Plastik %) ---
    df2_sel = df2[["Tahun", "Provinsi", "Kabupaten/Kota", "Plastik(%)"]].copy()
    df2_sel["Plastik(%)"] = clean_numeric(df2_sel["Plastik(%)"])
    df2_sel["Key_Prov"] = clean_location(df2_sel["Provinsi"])
    df2_sel["Key_Kab"] = clean_location(df2_sel["Kabupaten/Kota"])

    # --- FILE 3: Capaian Kinerja ---
    cols_f3 = [
        "Tahun",
        "Provinsi",
        "Kabupaten/Kota",
        "Timbulan Sampah Tahunan (ton/tahun)(A)",
        "Pengurangan Sampah Tahunan (ton/tahun)(B)",
        "Penanganan Sampah Tahunan (ton/tahun)(C)",
        "Sampah Terkelola Tahunan (ton/tahun)(B+C)",
        "Daur ulang Sampah Tahunan (ton/tahun)(D)",
        "Recycling Rate(D+E)/A",
    ]
    df3_sel = df3[cols_f3].copy()
    for col in cols_f3[3:]:
        df3_sel[col] = clean_numeric(df3_sel[col])
    df3_sel["Key_Prov"] = clean_location(df3_sel["Provinsi"])
    df3_sel["Key_Kab"] = clean_location(df3_sel["Kabupaten/Kota"])

    # --- FILE 4: TPS 3R / UPS ---
    df4["Jenis"] = df4["Jenis"].astype(str)
    df4_filtered = df4[df4["Jenis"].str.contains("TPS 3R|UPS", case=False, na=False)].copy()
    df4_agg = (
        df4_filtered.groupby(["Tahun", "Provinsi", "Kabupaten/Kota"])
        .agg({"Sampahmasuk (ton/thn)": "sum", "Sampahterkelola (ton/thn)": "sum"})
        .reset_index()
    )
    df4_agg["Key_Prov"] = clean_location(df4_agg["Provinsi"])
    df4_agg["Key_Kab"] = clean_location(df4_agg["Kabupaten/Kota"])
    df4_agg.rename(
        columns={
            "Sampahmasuk (ton/thn)": "TPS3R_SampahMasuk",
            "Sampahterkelola (ton/thn)": "TPS3R_SampahTerkelola",
        },
        inplace=True,
    )

    # --- FILE 5: TPA / TPST ---
    df5_agg = (
        df5.groupby(["Tahun", "Provinsi", "Kabupaten/Kota"])
        .agg({"Sampahmasuk (ton/thn)": "sum", "Sampahmasuk Landfill (ton/thn)": "sum"})
        .reset_index()
    )
    df5_agg["Key_Prov"] = clean_location(df5_agg["Provinsi"])
    df5_agg["Key_Kab"] = clean_location(df5_agg["Kabupaten/Kota"])
    df5_agg.rename(
        columns={
            "Sampahmasuk (ton/thn)": "TPA_SampahMasuk",
            "Sampahmasuk Landfill (ton/thn)": "TPA_SampahLandfill",
        },
        inplace=True,
    )

    # --- FILE 6: Penduduk (Dukcapil) ---
    df6_sel = df6[["tahun", "prov", "kabkot", "jumlah_penduduk"]].copy()
    df6_sel.rename(columns={"tahun": "Tahun", "jumlah_penduduk": "Penduduk"}, inplace=True)
    df6_sel["Penduduk"] = clean_numeric(df6_sel["Penduduk"])
    df6_sel["Key_Prov"] = clean_location(df6_sel["prov"])
    df6_sel["Key_Kab"] = clean_location(df6_sel["kabkot"])
    # Kalau ada lebih dari satu baris per tahun/prov/kab (semester 1 & 2), ambil penduduk terbesar
    df6_sel = df6_sel.sort_values("Penduduk", ascending=False).drop_duplicates(
        ["Tahun", "Key_Prov", "Key_Kab"]
    )

    return {
        "df1_sel": df1_sel,
        "df2_sel": df2_sel,
        "df3_sel": df3_sel,
        "df4_agg": df4_agg,
        "df5_agg": df5_agg,
        "df6_sel": df6_sel,
    }


# =============================
# 4. MERGE + FEATURE ENGINEERING
# =============================
def merge_and_feature_engineer(pre: dict):
    """
    Gabungkan semua file menjadi:
    - merged  : hasil join besar
    - final_df: tabel akhir (belum one-hot, belum scaling)
    """
    df1_sel = pre["df1_sel"]
    df2_sel = pre["df2_sel"]
    df3_sel = pre["df3_sel"]
    df4_agg = pre["df4_agg"]
    df5_agg = pre["df5_agg"]
    df6_sel = pre["df6_sel"]

    merged = df1_sel.copy()

    # File 2 - Plastik %
    merged = merged.merge(
        df2_sel[["Tahun", "Key_Prov", "Key_Kab", "Plastik(%)"]],
        on=["Tahun", "Key_Prov", "Key_Kab"],
        how="left",
    )

    # File 3 - Capaian (tanpa nama prov/kab)
    merged = merged.merge(
        df3_sel.drop(columns=["Provinsi", "Kabupaten/Kota"]),
        on=["Tahun", "Key_Prov", "Key_Kab"],
        how="left",
    )

    # File 4 - TPS 3R
    merged = merged.merge(
        df4_agg[["Tahun", "Key_Prov", "Key_Kab", "TPS3R_SampahMasuk", "TPS3R_SampahTerkelola"]],
        on=["Tahun", "Key_Prov", "Key_Kab"],
        how="left",
    )

    # File 5 - TPA
    merged = merged.merge(
        df5_agg[["Tahun", "Key_Prov", "Key_Kab", "TPA_SampahMasuk", "TPA_SampahLandfill"]],
        on=["Tahun", "Key_Prov", "Key_Kab"],
        how="left",
    )

    # File 6 - Penduduk
    merged = merged.merge(
        df6_sel[["Tahun", "Key_Prov", "Key_Kab", "Penduduk"]],
        on=["Tahun", "Key_Prov", "Key_Kab"],
        how="left",
    )

    # ---------- FEATURE ENGINEERING ----------
    merged["Estimasi_Timbulan_Plastik_Ton"] = (
        merged["Timbulan Sampah Tahunan(ton)"] * (merged["Plastik(%)"] / 100.0)
    )
    merged["Estimasi_DaurUlang_Plastik_Ton"] = (
        merged["Daur ulang Sampah Tahunan (ton/tahun)(D)"] * (merged["Plastik(%)"] / 100.0)
    )
    merged["Estimasi_Plastik_Terkelola_Ton"] = (
        merged["Sampah Terkelola Tahunan (ton/tahun)(B+C)"] * (merged["Plastik(%)"] / 100.0)
    )

    # Rename kolom capaian
    rename_map = {
        "Timbulan Sampah Tahunan (ton/tahun)(A)": "Capaian_Timbulan_A",
        "Pengurangan Sampah Tahunan (ton/tahun)(B)": "Capaian_Pengurangan_B",
        "Penanganan Sampah Tahunan (ton/tahun)(C)": "Capaian_Penanganan_C",
        "Sampah Terkelola Tahunan (ton/tahun)(B+C)": "Capaian_Terkelola_BC",
        "Daur ulang Sampah Tahunan (ton/tahun)(D)": "Capaian_DaurUlang_D",
        "Recycling Rate(D+E)/A": "Recycling_Rate",
    }
    final_df = merged.rename(columns=rename_map)

    ordered_cols = [
        "Tahun",
        "Provinsi",
        "Kabupaten/Kota",
        "Key_Prov",
        "Key_Kab",
        "Timbulan Sampah Tahunan(ton)",
        "Plastik(%)",
        "Capaian_Timbulan_A",
        "Capaian_Pengurangan_B",
        "Capaian_Penanganan_C",
        "Capaian_Terkelola_BC",
        "Capaian_DaurUlang_D",
        "Recycling_Rate",
        "Estimasi_Timbulan_Plastik_Ton",
        "Estimasi_DaurUlang_Plastik_Ton",
        "Estimasi_Plastik_Terkelola_Ton",
        "TPS3R_SampahMasuk",
        "TPS3R_SampahTerkelola",
        "TPA_SampahMasuk",
        "TPA_SampahLandfill",
        "Penduduk",
    ]
    ordered_cols = [c for c in ordered_cols if c in final_df.columns]
    final_df = final_df[ordered_cols]

    return merged, final_df


# =============================
# 5. ENCODING, OUTLIER, SCALING
# =============================
OUTLIER_COLS = [
    "Timbulan Sampah Tahunan(ton)",
    "Plastik(%)",
    "Estimasi_Timbulan_Plastik_Ton",
    "Capaian_Timbulan_A",
    "Capaian_Pengurangan_B",
    "Capaian_Penanganan_C",
    "Capaian_Terkelola_BC",
    "Capaian_DaurUlang_D",
    "Recycling_Rate",
    "Estimasi_DaurUlang_Plastik_Ton",
    "Estimasi_Plastik_Terkelola_Ton",
    "TPS3R_SampahMasuk",
    "TPS3R_SampahTerkelola",
    "TPA_SampahMasuk",
    "TPA_SampahLandfill",
    "Penduduk",
]


def encode_one_hot(final_df: pd.DataFrame) -> pd.DataFrame:
    df = final_df.copy()
    if "Provinsi" in df.columns:
        df = pd.get_dummies(df, columns=["Provinsi"], prefix="Provinsi")
    if "Kabupaten/Kota" in df.columns:
        df = pd.get_dummies(df, columns=["Kabupaten/Kota"], prefix="KabKota")
    return df


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    return df


def handle_outliers(df_encoded: pd.DataFrame) -> pd.DataFrame:
    df = df_encoded.copy()
    for col in OUTLIER_COLS:
        if col in df.columns:
            df = cap_outliers_iqr(df, col)
    return df


def scale_numeric(df_outlier: pd.DataFrame):
    df = df_outlier.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # kolom OHE + Tahun tidak di-scale
    exclude = [c for c in numeric_cols if c.startswith("Provinsi_") or c.startswith("KabKota_")]
    if TIME_COL in numeric_cols:
        exclude.append(TIME_COL)

    columns_to_scale = [c for c in numeric_cols if c not in exclude]

    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df, scaler, columns_to_scale


def build_full_data_pipeline(base_path="."):
    """
    Pipeline lengkap:
    Excel → cleaning per file → merge → feature engineering →
    one-hot → outlier → scaling.
    """
    raw = load_raw_data(base_path)
    pre = preprocess_each_file(raw)
    merged, final_df = merge_and_feature_engineer(pre)
    encoded = encode_one_hot(final_df)
    outlier = handle_outliers(encoded)
    final_data, scaler, columns_to_scale = scale_numeric(outlier)

    return {
        "raw": raw,
        "preprocessed": pre,
        "merged": merged,
        "final_df": final_df,
        "final_data": final_data,
        "scaler": scaler,
        "columns_to_scale": columns_to_scale,
    }


# =============================
# 6. MODEL HYBRID PROPHET–XGBOOST
# =============================
def evaluate(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    )

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE (%)": mape,
    }


def add_prophet_trend_feature(final_data: pd.DataFrame):
    """
    Tambahkan kolom 'prophet_trend' per Kab/Kota.
    Kalau Prophet tidak tersedia, isi 0 saja (supaya tetap jalan).
    """
    df = final_data.copy()
    df["prophet_trend"] = 0.0

    if not PROPHET_AVAILABLE:
        return df, False

    region_cols = [c for c in df.columns if c.startswith("KabKota_")]

    for region in region_cols:
        mask_region = df[region] == 1
        ts = (
            df.loc[mask_region, [TIME_COL, TARGET]]
            .dropna()
            .sort_values(TIME_COL)
            .drop_duplicates(TIME_COL)
        )
        if len(ts) < 3:
            continue

        ts["ds"] = pd.to_datetime(ts[TIME_COL].astype(int).astype(str), format="%Y")
        ts["y"] = ts[TARGET]

        model = Prophet(yearly_seasonality=False, changepoint_prior_scale=0.1)
        model.fit(ts[["ds", "y"]])

        forecast = model.predict(ts[["ds"]])
        trend_map = dict(zip(ts[TIME_COL], forecast["yhat"].values))

        idx = df[mask_region].index
        df.loc[idx, "prophet_trend"] = df.loc[idx, TIME_COL].map(trend_map)

    if df["prophet_trend"].isna().all():
        df["prophet_trend"] = 0.0
    else:
        df["prophet_trend"] = df["prophet_trend"].fillna(df["prophet_trend"].median())

    return df, True


def train_hybrid_model(final_data: pd.DataFrame):
    """
    Latih model hybrid Prophet–XGBoost.
    Output: dict berisi model, data train/test, dan hasil evaluasi.
    """
    # Filter tahun mulai START_YEAR
    df = final_data[final_data[TIME_COL] >= START_YEAR].copy()

    # Tambah fitur tren Prophet
    df_with_trend, prophet_used = add_prophet_trend_feature(df)

    # Buang kolom key string yang tidak dipakai model
    for col in ["Key_Prov", "Key_Kab"]:
        if col in df_with_trend.columns:
            df_with_trend.drop(columns=[col], inplace=True)

    # Split fitur & target
    X = df_with_trend.drop(columns=[TARGET])
    y = df_with_trend[TARGET]

    # Train-test split berbasis tahun
    X_train = X[df_with_trend[TIME_COL] <= TRAIN_END]
    X_test = X[df_with_trend[TIME_COL] > TRAIN_END]
    y_train = y[df_with_trend[TIME_COL] <= TRAIN_END]
    y_test = y[df_with_trend[TIME_COL] > TRAIN_END]

    # XGBoost Regressor
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics_train = evaluate(y_train, train_pred)
    metrics_test = evaluate(y_test, test_pred)

    results = pd.DataFrame(
        [
            ["Hybrid Prophet–XGBoost", "Train", metrics_train["R2"], metrics_train["RMSE"],
             metrics_train["MAE"], metrics_train["MAPE (%)"]],
            ["Hybrid Prophet–XGBoost", "Test", metrics_test["R2"], metrics_test["RMSE"],
             metrics_test["MAE"], metrics_test["MAPE (%)"]],
        ],
        columns=["Model", "Stage", "R2", "RMSE", "MAE", "MAPE (%)"],
    )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
        "final_with_trend": df_with_trend,
        "prophet_used": prophet_used,
    }


def get_feature_importances(model: XGBRegressor, X_train: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Ambil n fitur terpenting dari XGBoost."""
    importances = model.feature_importances_
    fi = (
        pd.DataFrame({"Fitur": X_train.columns, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )
    return fi.reset_index(drop=True)


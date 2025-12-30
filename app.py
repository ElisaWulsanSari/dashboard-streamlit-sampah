# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


from model import (
    build_full_data_pipeline,
    train_hybrid_model,
    get_feature_importances,
    TARGET,
    TIME_COL,
)

st.set_page_config(
    page_title="Dashboard Prediksi Timbulan Sampah Plastik",
    layout="wide",
    page_icon="♻️",
)
st.markdown("""
<style>
:root {
    --primary-dark: #1C274C;
    --primary-mid:  #2C4E8A;
    --primary-soft: #6DB6E1;
    --primary-light:#D4E7F7;
}

/* HEADER KEMBALI TRANSPARAN */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
header[data-testid="stHeader"] * {
    color: #1C274C !important;
}

/* BACKGROUND HALAMAN */
.block-container {
    padding-top: 0rem !important;
    background: linear-gradient(180deg, #ffffff 0%, var(--primary-light) 100%);
}

/* SIDEBAR GRADIENT MODERN */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--primary-dark), var(--primary-mid) 70%, var(--primary-soft));
    color: #ffffff;
    padding-top: 2.2rem;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #ffffff !important;
}

/* TITLE ICON */
.sidebar-title-custom span.icon {
    background: var(--primary-soft);
    color: #1C274C;
    font-weight: 900;
}

/* RADIO NAV */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(255,255,255,0.18);
    transform: translateX(3px);
}
section[data-testid="stSidebar"] .stRadio input[type="radio"] {
    accent-color: var(--primary-soft);
}

/* SIDEBAR FOOTER TITLE */
.sidebar-footer-title {
    margin-top: auto;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.3);
    text-align: center;
    font-size: 0.9rem;
    color: #ffffff;
    opacity: 0.9;
}
.sidebar-footer-title span {
    font-size: 0.82rem;
    font-weight: 400;
}

/* TOMBOL DATASET */
.dataset-btn {
    background: linear-gradient(135deg, var(--primary-mid), var(--primary-soft));
    border: none;
    border-radius: 18px;
    padding: 40px 60px;
    min-width: 380px;
    margin: 14px 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffffff;
    cursor: pointer;
    text-align: center;
    box-shadow: 0 10px 24px rgba(28, 39, 76, 0.28);
    transition: 0.18s ease-in-out;
}
.dataset-btn:hover {
    transform: translateY(-3px);
    background: linear-gradient(135deg, var(--primary-dark), var(--primary-mid));
    box-shadow: 0 16px 34px rgba(28,39,76,0.38);
}

/* ====== JUDUL DI BAWAH NAVIGASI LEBIH TURUN ====== */
.sidebar-title-custom {
    margin-bottom: 1.1rem !important;
}

.sidebar-footer-title {
    margin-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
:root {
    --primary-dark: #1C274C;
    --primary-mid:  #2C4E8A;
    --primary-soft: #6DB6E1;
}

div.stButton > button {
    background: linear-gradient(135deg, var(--primary-mid), var(--primary-soft));
    border: none;
    border-radius: 18px;
    padding: 40px 60px;
    min-width: 380px;
    margin: 14px 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffffff !important;
    cursor: pointer;
    text-align: center;
    box-shadow: 0 10px 24px rgba(28, 39, 76, 0.28);
    transition: 0.18s ease-in-out;
}
div.stButton > button:hover {
    transform: translateY(-3px);
    background: linear-gradient(135deg, var(--primary-dark), var(--primary-mid));
    box-shadow: 0 16px 34px rgba(28,39,76,0.38);
}
</style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* TOMBOL DATASET BERANDA + CLEANING */
    .dataset-grid div.stButton > button,
    .clean-grid div.stButton > button {
        background: linear-gradient(135deg, var(--primary-mid), var(--primary-soft));
        border: none;
        border-radius: 18px;
        padding: 26px 24px;
        width: 100%;              /* biar isi kolom penuh */
        margin: 10px 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff !important;
        cursor: pointer;
        text-align: center;
        box-shadow: 0 10px 24px rgba(28, 39, 76, 0.28);
        transition: 0.18s ease-in-out;
        white-space: normal;      /* supaya teks bisa 2 baris kalau kepanjangan */
    }

    .dataset-grid div.stButton > button:hover,
    .clean-grid div.stButton > button:hover {
        transform: translateY(-3px);
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-mid));
        box-shadow: 0 16px 34px rgba(28,39,76,0.38);
    }

    .clean-grid div.stButton > button:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)








# ============================
# CACHE DATA & MODEL
# ============================
@st.cache_data
def load_pipeline():
    # Folder kerja saat run streamlit
    return build_full_data_pipeline(".")


@st.cache_resource
def load_model():
    pipe = build_full_data_pipeline(".")
    return train_hybrid_model(pipe["final_data"])


pipe = load_pipeline()
model_bundle = load_model()

raw = pipe["raw"]
pre = pipe["preprocessed"]
merged = pipe["merged"]
final_df = pipe["final_df"]
final_data = pipe["final_data"]
scaler = pipe["scaler"]
columns_to_scale = pipe["columns_to_scale"]

model = model_bundle["model"]
X_train = model_bundle["X_train"]
X_test = model_bundle["X_test"]
y_train = model_bundle["y_train"]
y_test = model_bundle["y_test"]
results = model_bundle["results"]
final_with_trend = model_bundle["final_with_trend"]
prophet_used = model_bundle["prophet_used"]


# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title-custom">
            <span class="icon">⚙️</span>
            <span>Navigasi</span>
        </div>
        <p class="sidebar-subtitle">Pilih halaman utama:</p>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "",
        [
            "Beranda",
            "Cleaning & Penggabungan",
            "EDA",
            "Modeling & Evaluasi",
            "Prediksi Manual",
        ],
    )

    # Footer judul penelitian di bagian paling bawah sidebar
    st.markdown(
        """
        <div class="sidebar-footer-title">
            Prediksi Timbulan Sampah Plastik
            <span>
                Berdasarkan Kinerja Infrastruktur Daur Ulang dan Inovasi Industri Hijau
                Menggunakan Pendekatan Hybrid Prophet–XGBoost
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================
# 1. OVERVIEW + DATA MENTAH
# ============================
if page == "Beranda":
    st.title("♻️ Dashboard Prediksi Timbulan Sampah Plastik")

    # ===== Ringkasan angka penting =====
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Baris Final", f"{len(final_df):,}")
    with col2:
        st.metric("Kab/Kota Unik", f"{final_df['Key_Kab'].nunique():,}")
    with col3:
        st.metric(
            "Rentang Tahun",
            f"{int(final_df[TIME_COL].min())}–{int(final_df[TIME_COL].max())}",
        )

    # ===== Pilihan Dataset Mentah =====
    st.markdown("---")

    dataset_map = {
        "FILE1": ("Timbulan Sampah", raw["df1_timbulan"]),
        "FILE2": ("Komposisi Jenis Sampah", raw["df2_komposisi"]),
        "FILE3": ("Capaian Kinerja", raw["df3_capaian"]),
        "FILE4": ("Data TPS 3R / UPS", raw["df4_tps3r"]),
        "FILE5": ("Fasilitas TPST/TPA", raw["df5_tpatpst"]),
        "FILE6": ("Penduduk Dukcapil", raw["df6_penduduk"]),
    }

        # default pilihan dataset
    if "selected_dataset" not in st.session_state:
        st.session_state["selected_dataset"] = "FILE1"

    st.subheader("Dataset Mentah")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Timbulan Sampah"):
            st.session_state["selected_dataset"] = "FILE1"
    with col2:
        if st.button("Komposisi Sampah"):
            st.session_state["selected_dataset"] = "FILE2"
    with col3:
        if st.button("Capaian Kinerja"):
            st.session_state["selected_dataset"] = "FILE3"

    st.write(""); st.write("")

    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("TPS 3R / UPS"):
            st.session_state["selected_dataset"] = "FILE4"
    with col5:
        if st.button("TPST / TPA"):
            st.session_state["selected_dataset"] = "FILE5"
    with col6:
        if st.button("Penduduk"):
            st.session_state["selected_dataset"] = "FILE6"


    st.markdown("---")

    # ===== Preview Data Mentah =====
    key = st.session_state["selected_dataset"]
    label, df_show = dataset_map[key]

    st.subheader(f"Preview Data: {label}")
    st.dataframe(df_show.head(12), use_container_width=True)
    st.caption("*Menampilkan 12 baris pertama*")



    


# ============================
# 3. CLEANING & PENGGABUNGAN
# ============================
elif page == "Cleaning & Penggabungan":
    st.title("Cleaning & Penggabungan Data")

    # Hanya 2 tab: cleaning & data gabungan
    tab1, tab2 = st.tabs(
        ["Cleaning per File", "Data Gabungan Lengkap"]
    )

    # ----------------------------
    # TAB 1 — CLEANING PER FILE
    # ----------------------------
    with tab1:
        st.subheader("Cleaning File")

        # Mapping dataframe hasil cleaning
        clean_map = {
            "FILE1": ("Timbulan", pre["df1_sel"]),
            "FILE2": ("Komposisi Plastik", pre["df2_sel"]),
            "FILE3": ("Capaian Kinerja", pre["df3_sel"]),
            "FILE4": ("TPS3R agregat", pre["df4_agg"]),
            "FILE5": ("TPA agregat", pre["df5_agg"]),
            "FILE6": ("Penduduk ", pre["df6_sel"]),
        }

        # default pilihan cleaning
        if "selected_clean" not in st.session_state:
            st.session_state["selected_clean"] = "FILE1"

        # grid tombol 6 kolom dalam 1 baris
        clean_container = st.container()
        clean_container.markdown('<div class="clean-grid">', unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = clean_container.columns(6)

        with c1:
            if st.button("Cleaning Timbulan", key="clean_f1"):
                st.session_state["selected_clean"] = "FILE1"
        with c2:
            if st.button("Cleaning Komposisi", key="clean_f2"):
                st.session_state["selected_clean"] = "FILE2"
        with c3:
            if st.button("Cleaning Capaian", key="clean_f3"):
                st.session_state["selected_clean"] = "FILE3"
        with c4:
            if st.button("Cleaning TPS 3R", key="clean_f4"):
                st.session_state["selected_clean"] = "FILE4"
        with c5:
            if st.button("Cleaning TPST / TPA", key="clean_f5"):
                st.session_state["selected_clean"] = "FILE5"
        with c6:
            if st.button("Cleaning Penduduk", key="clean_f6"):
                st.session_state["selected_clean"] = "FILE6"

        clean_container.markdown("</div>", unsafe_allow_html=True)

        # Tampilkan dataframe sesuai tombol yang dipilih
        key_clean = st.session_state["selected_clean"]
        label_clean, df_clean = clean_map[key_clean]

        st.markdown("---")
        st.subheader(f"Preview Hasil Cleaning: {label_clean}")
        st.dataframe(df_clean.head(30), use_container_width=True)
        st.caption("*Menampilkan 30 baris pertama*")

    # ----------------------------
    # TAB 2 — DATA GABUNGAN LENGKAP
    # ----------------------------
    with tab2:
        st.subheader("Data Gabungan 6 File (Final Merge)")
        st.markdown(
            """
            Tabel berikut adalah hasil penggabungan seluruh file
            menggunakan key `Tahun + Key_Prov + Key_Kab` setelah proses cleaning.
            """
        )

        # Misal kamu sudah load di awal:
        df_gabungan = pd.read_csv("Data_Sampah_Gabungan_Lengkap (1).csv")
        st.dataframe(df_gabungan.head(30), use_container_width=True)
        st.caption("*Menampilkan 30 baris pertama*")


# ================================
# 4. DATA AKHIR & EDA
# ================================
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.ticker as ticker

elif page == "EDA":  # pastikan string-nya SAMA dengan di sidebar, tanpa spasi
    # ===================== HEADER CARD =====================
    st.title("Eksplorasi Data (EDA)")

    st.markdown("## visualisasi permasalahan timbulan sampah plastik di Indonesia ")



    # -------- Sumber data: pakai final_df yang sudah dibersihkan --------
    eda_df = final_df.copy()
    final_data = final_df.copy()

    # -------- 1. Bersihkan kolom Tahun & filter periode --------
    if 'Tahun' not in eda_df.columns:
        st.error("Kolom 'Tahun' tidak ditemukan di final_df. Cek kembali nama kolomnya.")
        st.stop()

    eda_df['Tahun'] = pd.to_numeric(eda_df['Tahun'], errors='coerce')
    eda_df = eda_df.dropna(subset=['Tahun'])
    eda_df['Tahun'] = eda_df['Tahun'].astype(int)

    df_analisis = eda_df[eda_df['Tahun'] >= 2019].copy()
    df_period = eda_df[(eda_df['Tahun'] >= 2019) & (eda_df['Tahun'] <= 2024)].copy()

    # =====================================================================
    # 2. Tren Timbulan Sampah Plastik Nasional (2019–2024)
    # =====================================================================
    if 'Estimasi_Timbulan_Plastik_Ton' not in df_analisis.columns:
        st.error("Kolom 'Estimasi_Timbulan_Plastik_Ton' tidak ada di data. Cek kembali final_df.")
    else:
        nasional_trend = (
            df_analisis
            .groupby('Tahun')['Estimasi_Timbulan_Plastik_Ton']
            .sum()
            .reset_index()
        )
        nasional_trend['Growth_Rate'] = nasional_trend['Estimasi_Timbulan_Plastik_Ton'].pct_change() * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=nasional_trend,
            x='Tahun',
            y='Estimasi_Timbulan_Plastik_Ton',
            marker='o',
            color='navy',
            linewidth=2.5,
            ax=ax
        )

        for i in range(len(nasional_trend)):
            year = nasional_trend['Tahun'].iloc[i]
            val = nasional_trend['Estimasi_Timbulan_Plastik_Ton'].iloc[i]
            growth = nasional_trend['Growth_Rate'].iloc[i]

            if pd.isna(growth):
                label = f"{val/1e6:.2f} Jt Ton"
            else:
                trend_symbol = "▲" if growth > 0 else "▼"
                label = f"{val/1e6:.2f} Jt Ton\n({trend_symbol}{abs(growth):.1f}%)"

            ax.text(year, val, label, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title('Tren Timbulan Sampah Plastik Nasional (2019–2024)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Estimasi Timbulan Plastik (Ton)')
        ax.set_xlabel('Tahun')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        st.pyplot(fig)

    # =====================================================================
    # 3. Dinamika Tren Regional: Top 5 Provinsi
    # =====================================================================
    if not all(col in df_analisis.columns for col in ['Provinsi', 'Estimasi_Timbulan_Plastik_Ton']):
        st.warning("Kolom 'Provinsi' atau 'Estimasi_Timbulan_Plastik_Ton' tidak lengkap untuk analisis regional.")
    else:
        top_provs = (
            df_analisis
            .groupby('Provinsi')['Estimasi_Timbulan_Plastik_Ton']
            .sum()
            .nlargest(5)
            .index
            .tolist()
        )

        prov_trend = (
            df_analisis[df_analisis['Provinsi'].isin(top_provs)]
            .groupby(['Tahun', 'Provinsi'])['Estimasi_Timbulan_Plastik_Ton']
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=prov_trend,
            x='Tahun',
            y='Estimasi_Timbulan_Plastik_Ton',
            hue='Provinsi',
            marker='o',
            linewidth=2,
            palette='tab10',
            ax=ax
        )

        ax.set_title('Dinamika Tren Regional: Top 5 Provinsi (2019–2024)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Estimasi Timbulan Plastik (Ton)')
        ax.set_xlabel('Tahun')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
        ax.legend(title='Provinsi', loc='upper left')
        st.pyplot(fig)

    # =====================================================================
    # 4. BUKTI 1 & 2: Skala vs Konsumsi Per Kapita
    # =====================================================================
    if 'Penduduk' not in df_period.columns:
        st.warning("Kolom 'Penduduk' tidak ada, analisis per kapita dilewati.")
    else:
        df_period['Plastik_Per_Kapita_Kg'] = (
            df_period['Estimasi_Timbulan_Plastik_Ton'] * 1000
        ) / df_period['Penduduk'].replace(0, np.nan)

        prov_stats = (
            df_period
            .groupby('Provinsi')
            .agg({
                'Estimasi_Timbulan_Plastik_Ton': 'sum',
                'Plastik_Per_Kapita_Kg': 'mean'
            })
            .reset_index()
        )

        top_total = prov_stats.sort_values('Estimasi_Timbulan_Plastik_Ton', ascending=False).head(10)
        top_capita = prov_stats.sort_values('Plastik_Per_Kapita_Kg', ascending=False).head(10)

        fig, axes = plt.subplots(2, 1, figsize=(12, 14))
        plt.subplots_adjust(hspace=0.4)

        # BUKTI 1: Masalah skala
        sns.barplot(
            data=top_total,
            x='Estimasi_Timbulan_Plastik_Ton',
            y='Provinsi',
            ax=axes[0],
            palette='Blues_r'
        )
        axes[0].set_title(
            'BUKTI 1: Masalah Skala (Akumulasi 2019–2024)\nWilayah Penduduk Besar = Volume Sampah Besar',
            fontsize=14, fontweight='bold', loc='left'
        )
        axes[0].set_xlabel('Total Akumulasi Timbulan Plastik (Ton)')
        axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f} Jt'.format(x/1e6)))

        for i, v in enumerate(top_total['Estimasi_Timbulan_Plastik_Ton']):
            label = f"{v/1e6:.1f} Jt" if v > 1e6 else f"{v/1000:,.0f} Rb"
            axes[0].text(v, i, f" {label} Ton", va='center', fontweight='bold')

        # BUKTI 2: Masalah konsumsi
        sns.barplot(
            data=top_capita,
            x='Plastik_Per_Kapita_Kg',
            y='Provinsi',
            ax=axes[1],
            palette='Oranges_r'
        )
        axes[1].set_title(
            'BUKTI 2: Masalah Konsumsi (Rata-rata 2019–2024)\nWilayah Populasi Kecil Tapi Intensitas Tinggi',
            fontsize=14, fontweight='bold', loc='left'
        )
        axes[1].set_xlabel('Rata-rata Timbulan per Orang (kg/tahun)')

        for i, v in enumerate(top_capita['Plastik_Per_Kapita_Kg']):
            axes[1].text(v, i, f" {v:.1f} kg", va='center', fontweight='bold')

        st.pyplot(fig)

    # =====================================================================
    # 5. REALITA 1 & 2: Daur Ulang vs Residu (Bar + Donut)
    # =====================================================================
    if 'Estimasi_DaurUlang_Plastik_Ton' not in df_period.columns:
        st.warning("Kolom 'Estimasi_DaurUlang_Plastik_Ton' tidak ada, analisis daur ulang dilewati.")
    else:
        prov_gap = (
            df_period
            .groupby('Provinsi')
            .agg({
                'Estimasi_Timbulan_Plastik_Ton': 'sum',
                'Estimasi_DaurUlang_Plastik_Ton': 'sum'
            })
            .reset_index()
        )

        top_5 = prov_gap.sort_values('Estimasi_Timbulan_Plastik_Ton', ascending=False).head(5)
        top_5_melt = top_5.melt(
            id_vars='Provinsi',
            value_vars=['Estimasi_Timbulan_Plastik_Ton', 'Estimasi_DaurUlang_Plastik_Ton'],
            var_name='Jenis',
            value_name='Ton'
        )
        top_5_melt['Juta_Ton'] = top_5_melt['Ton'] / 1e6

        fig, axes = plt.subplots(2, 1, figsize=(12, 14))
        plt.subplots_adjust(hspace=0.4)

        # REALITA 1: barplot
        sns.barplot(
            data=top_5_melt,
            x='Juta_Ton',
            y='Provinsi',
            hue='Jenis',
            ax=axes[0],
            palette=['#e74c3c', '#2ecc71']
        )
        axes[0].set_title(
            'REALITA 1: Jauh Panggang dari Api (Top 5 Provinsi)\nTotal Sampah (Merah) vs Kapasitas Daur Ulang (Hijau)',
            fontsize=14, fontweight='bold', loc='left'
        )
        axes[0].set_xlabel('Volume (Juta Ton)')
        axes[0].set_ylabel('')
        axes[0].grid(axis='x', linestyle='--', alpha=0.5)
        axes[0].legend(
            ['Total Sampah (Masalah)', 'Kapasitas Daur Ulang (Solusi)'],
            loc='lower right'
        )

        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.1f Jt', padding=3, fontweight='bold', fontsize=10)

        # REALITA 2: donut nasional
        total_waste = df_period['Estimasi_Timbulan_Plastik_Ton'].sum()
        total_recycle = df_period['Estimasi_DaurUlang_Plastik_Ton'].sum()
        residue = total_waste - total_recycle
        recycle_rate = (total_recycle / total_waste) * 100 if total_waste > 0 else 0

        sizes = [total_recycle, residue]
        labels = ['Daur Ulang', 'Residu (Tertimbun)']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)

        wedges, texts, autotexts = axes[1].pie(
            sizes,
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            colors=colors,
            explode=explode
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[1].add_artist(centre_circle)

        axes[1].set_title(
            'REALITA 2: Potret Nasional (2019–2024)\nMayoritas Sampah Plastik Berakhir Menjadi Residu',
            fontsize=14, fontweight='bold'
        )
        axes[1].legend(
            wedges,
            labels,
            loc='upper right',
            bbox_to_anchor=(1, 0.9),
            title='Status Sampah'
        )
        axes[1].text(
            0, 0,
            f'Hanya\n{recycle_rate:.1f}%\nTerdaur Ulang',
            ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2ecc71'
        )

        st.pyplot(fig)

    # =====================================================================
    # 6. Heatmap Korelasi Numerik (final_data)
    # =====================================================================
    st.subheader("Heatmap Korelasi Fitur Numerik (Final Data)")

    numeric_cols_for_heatmap = final_data.select_dtypes(include=np.number).columns.tolist()
    cols_to_exclude = ['Tahun'] + [
        col for col in numeric_cols_for_heatmap
        if col.startswith(('Provinsi_', 'KabKota_'))
    ]
    numeric_cols_for_heatmap = [
        col for col in numeric_cols_for_heatmap
        if col not in cols_to_exclude
    ]

    if len(numeric_cols_for_heatmap) == 0:
        st.warning("Tidak ada kolom numerik utama yang bisa dihitung korelasinya.")
    else:
        correlation_matrix = final_data[numeric_cols_for_heatmap].corr()

        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5,
            ax=ax
        )
        ax.set_title('Heatmap Korelasi Fitur Numerik Utama di Final Data', fontsize=18, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)

    st.markdown('</div></div>', unsafe_allow_html=True)



# ============================
# 5. MODELING & EVALUASI
# ============================
elif page == "Modeling & Evaluasi":
    # ========================= HEADER =========================
    st.title("Modeling & Evaluasi Hybrid Prophet–XGBoost")
    
    # ====================== HASIL EVALUASI =====================
    st.subheader("Hasil Evaluasi")
    st.dataframe(results, use_container_width=True)

    if not prophet_used:
        st.warning(
            "Library Prophet tidak tersedia di environment, sehingga fitur `prophet_trend` "
            "diisi 0 (model tetap berjalan, hanya tanpa komponen tren time series)."
        )

    # =================== AKTUAL vs PREDIKSI ====================
    st.subheader("Aktual vs Prediksi (Data Uji)")

    # Pastikan ada y_test dan prediksi untuk test
    # Jika y_pred_test belum dibuat di bagian modeling, lakukan prediksi di sini
    try:
        y_true = y_test
    except NameError:
        st.error("Variabel y_test belum didefinisikan. Pastikan proses modeling membuat y_test.")
        y_true = None
    y_pred_test = model.predict(X_test)

    if y_true is not None:
        try:
            y_pred_plot = y_pred_test
        except NameError:
            # Jika belum ada, prediksi ulang dari model
            y_pred_plot = model.predict(X_test)

        # Plot scatter seperti di contoh gambar
        fig_scatter, ax = plt.subplots(figsize=(9, 3))
        ax.scatter(y_true, y_pred_plot, alpha=0.6)

        # Garis diagonal (perfect prediction)
        min_val = min(float(y_true.min()), float(y_pred_plot.min()))
        max_val = max(float(y_true.max()), float(y_pred_plot.max()))
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        ax.set_xlabel("Nilai Aktual")
        ax.set_ylabel("Nilai Prediksi")
        ax.set_title("Aktual vs Prediksi (Data Uji)")
        ax.grid(True, linestyle="--", alpha=0.4)

        st.pyplot(fig_scatter)

# ============================
# 6. PREDIKSI MANUAL
# ============================
elif page == "Prediksi Manual":
    st.title("Prediksi Manual Timbulan Sampah Plastik")

    st.markdown(
        """
        - Pilih **tahun**, **provinsi**, dan **kabupaten/kota**  
        - Isi nilai fitur numerik (dalam skala 0–1, sudah dinormalisasi MinMax)  
        - Model akan mengembalikan **estimasi timbulan sampah plastik (ton/tahun)**  
        """
    )

    feature_cols = list(X_train.columns)

    prov_cols = [c for c in feature_cols if c.startswith("Provinsi_")]
    kab_cols = [c for c in feature_cols if c.startswith("KabKota_")]

    # Fitur numerik yang di-scale (kecuali target, kita tidak input manual)
    numeric_cols = [c for c in feature_cols if c in columns_to_scale and c != TARGET and c != TIME_COL]

    # Default = rata-rata nilai di data train (skala 0–1)
    default_numeric = X_train[numeric_cols].mean().to_dict()

    with st.form("manual_prediction"):
        col_tahun, col_prov, col_kab = st.columns(3)
        with col_tahun:
            tahun = st.number_input(
                "Tahun prediksi", min_value=2019, max_value=2035, value=2025, step=1
            )
        with col_prov:
            prov_label = st.selectbox(
                "Pilih Provinsi",
                sorted([c.replace("Provinsi_", "") for c in prov_cols]),
            )
        with col_kab:
            kab_label = st.selectbox(
                "Pilih Kab/Kota",
                sorted([c.replace("KabKota_", "") for c in kab_cols]),
            )

        st.markdown("### Input Fitur Numerik (skala 0–1 hasil MinMaxScaler)")
        cols = st.columns(3)
        numeric_inputs = {}
        for i, col in enumerate(numeric_cols):
            with cols[i % 3]:
                numeric_inputs[col] = st.number_input(
                    col,
                    min_value=0.0,
                    max_value=1.0,
                    value=float(default_numeric.get(col, 0.0)),
                    step=0.01,
                )

        submitted = st.form_submit_button("🔮 Prediksi")

    if submitted:
        # 1) Siapkan satu baris fitur
        input_data = {c: 0.0 for c in feature_cols}
        input_data[TIME_COL] = tahun

        prov_col_name = "Provinsi_" + prov_label
        kab_col_name = "KabKota_" + kab_label

        if prov_col_name in input_data:
            input_data[prov_col_name] = 1.0
        if kab_col_name in input_data:
            input_data[kab_col_name] = 1.0

        for col, val in numeric_inputs.items():
            input_data[col] = float(val)

        input_df = pd.DataFrame([input_data])

        # 2) Isi fitur prophet_trend berbasis historis Kab/Kota
        df_trend = final_with_trend
        if kab_col_name in df_trend.columns:
            trend_series = df_trend.loc[
                (df_trend[kab_col_name] == 1) & (df_trend[TIME_COL] <= tahun),
                "prophet_trend",
            ].dropna()
        else:
            trend_series = pd.Series(dtype=float)

        if not trend_series.empty:
            trend_val = float(trend_series.iloc[-1])
        else:
            global_trend = df_trend.loc[df_trend[TIME_COL] <= tahun, "prophet_trend"].dropna()
            trend_val = float(global_trend.mean()) if not global_trend.empty else 0.0

        if "prophet_trend" in input_df.columns:
            input_df.loc[0, "prophet_trend"] = trend_val

        # 3) Prediksi (hasil masih skala 0–1 karena target sudah di-scale)
        pred_scaled = float(model.predict(input_df)[0])

        # 4) Inverse transform ke satuan ton
        dummy = pd.DataFrame(
            np.zeros((1, len(columns_to_scale))), columns=columns_to_scale
        )
        dummy[TARGET] = pred_scaled
        restored = scaler.inverse_transform(dummy)[0]
        target_idx = columns_to_scale.index(TARGET)
        pred_ton = restored[target_idx]

        st.success("Prediksi berhasil dihitung!")
        st.metric("Estimasi Timbulan Sampah Plastik (ton/tahun)", f"{pred_ton:,.2f}")
        st.caption(
            "Catatan: fitur numerik diinput dalam skala 0–1 (MinMaxScaler), "
            "namun hasil akhir sudah dikembalikan ke satuan ton/tahun."
        )

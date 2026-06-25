import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Kalkulator Variabel FCM",
    page_icon="🧮",
    layout="wide"
)

# =========================================================
# CSS / STYLING
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1100px;
}

#MainMenu, footer {
    visibility: hidden;
}

:root {
    --accent: #4f46e5;
    --accent-light: #eef2ff;
    --border: #e5e7eb;
    --text: #111827;
    --muted: #6b7280;
    --green-bg: #f0fdf4;
    --green-border: #bbf7d0;
}

h1, h2, h3, h4 {
    color: var(--text);
    font-weight: 700;
}

.section-box {
    background: #f9fafb;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
}

.result-box {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 12px;
}

.result-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #e0e7ff;
    font-size: 0.92rem;
}

.result-row:last-child {
    border-bottom: none;
}

.result-label {
    color: #374151;
    font-weight: 500;
}

.result-value {
    color: #4f46e5;
    font-weight: 700;
    font-size: 0.98rem;
}

.pill-year {
    display: inline-block;
    background: #4f46e5;
    color: white;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 8px;
}

.avg-box {
    background: var(--green-bg);
    border: 1px solid var(--green-border);
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 10px;
}

.small-note {
    color: var(--muted);
    font-size: 0.84rem;
}

[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px;
}

.stButton button {
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stDownloadButton button {
    background: #111827;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.delete-btn button {
    background: #dc2626 !important;
    color: white !important;
}

.secondary-btn button {
    background: #111827 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "periods" not in st.session_state:
    st.session_state["periods"] = []

if "ticker_name" not in st.session_state:
    st.session_state["ticker_name"] = ""

if "fcm_input_list" not in st.session_state:
    st.session_state["fcm_input_list"] = []

# FIX #2: State untuk reset form
if "form_reset_counter" not in st.session_state:
    st.session_state["form_reset_counter"] = 0

# =========================================================
# KONSTANTA
# =========================================================
FEATURES = ['ROA', 'ROE', 'NPM', 'DER', 'NPL', 'Volatilitas', 'VaR', 'EPS']
SATUAN = {
    'ROA': '%',
    'ROE': '%',
    'NPM': '%',
    'DER': '%',
    'NPL': '%',
    'Volatilitas': '%',
    'VaR': '%',
    'EPS': 'Rp'
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def fmt_num(x, decimals=4):
    """Format angka dengan pemisah ribuan."""
    try:
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return str(x)

def parse_harga_list(harga_raw: str):
    """
    Parsing string harga saham bulanan:
    contoh: '4500, 4600, 4450'
    menjadi list float.
    """
    if not harga_raw or not harga_raw.strip():
        return []

    try:
        harga_list = [float(x.strip()) for x in harga_raw.split(",") if x.strip()]
    except Exception:
        raise ValueError("Format harga saham tidak valid. Gunakan angka dipisah koma.")

    if any(h <= 0 for h in harga_list):
        raise ValueError("Semua harga saham harus lebih besar dari 0.")

    return harga_list

def hitung_variabel(
    laba,
    pendapatan,
    aset,
    ekuitas,
    utang,
    npl_gross,
    total_kredit,
    saham_beredar,
    harga_list
):
    """
    Menghitung variabel FCM:
    - ROA = laba / aset * 100
    - ROE = laba / ekuitas * 100
    - NPM = laba / pendapatan * 100
    - DER = utang / ekuitas * 100
    - NPL = npl_gross / total_kredit * 100
    - EPS = laba (juta Rp -> Rp) / saham beredar
    - Volatilitas = std return bulanan * sqrt(12) * 100
    - VaR = Parametric VaR return bulanan 95% (%)
    """
    hasil = {}

    # Rasio keuangan
    hasil['ROA'] = round((laba / aset * 100), 4) if aset > 0 else 0.0
    hasil['ROE'] = round((laba / ekuitas * 100), 4) if ekuitas > 0 else 0.0
    hasil['NPM'] = round((laba / pendapatan * 100), 4) if pendapatan > 0 else 0.0
    hasil['DER'] = round((utang / ekuitas * 100), 4) if ekuitas > 0 else 0.0
    hasil['NPL'] = round((npl_gross / total_kredit * 100), 4) if total_kredit > 0 else 0.0
    hasil['EPS'] = round((laba * 1_000_000 / saham_beredar), 4) if saham_beredar > 0 else 0.0

    # Default
    hasil['Volatilitas'] = 0.0
    hasil['VaR'] = 0.0
    hasil['Jumlah_Data_Harga'] = len(harga_list)

    # Hitung risiko pasar jika data harga cukup
    if len(harga_list) >= 3:
        prices = np.array(harga_list, dtype=float)

        # Return bulanan sederhana
        returns = np.diff(prices) / prices[:-1]

        # Volatilitas bulanan
        vol_bulanan = np.std(returns, ddof=1) if len(returns) > 1 else 0.0

        # Annualized volatility
        vol_tahunan = vol_bulanan * np.sqrt(12) * 100

        # Parametric VaR return bulanan 95%
        mean_ret = np.mean(returns) if len(returns) > 0 else 0.0
        var_95 = max(0, (1.645 * vol_bulanan - mean_ret) * 100)

        hasil['Volatilitas'] = round(vol_tahunan, 4)
        hasil['VaR'] = round(var_95, 4)

    return hasil

def periods_to_dataframe(periods):
    """Mengubah list periods menjadi DataFrame terurut."""
    if not periods:
        return pd.DataFrame()

    df = pd.DataFrame(periods).sort_values("Tahun").reset_index(drop=True)
    return df

def create_excel_file(df_periods=None, df_fcm=None):
    """Membuat file Excel dalam memory untuk di-download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if df_periods is not None and not df_periods.empty:
            df_periods.to_excel(writer, index=False, sheet_name="Hasil_Per_Periode")
        if df_fcm is not None and not df_fcm.empty:
            df_fcm.to_excel(writer, index=False, sheet_name="Input_FCM")
    output.seek(0)
    return output

# =========================================================
# HEADER
# =========================================================
st.markdown("## 🧮 Kalkulator Variabel FCM")
st.caption(
    "Input angka mentah dari laporan keuangan → otomatis hitung ROA, ROE, NPM, DER, NPL, EPS. "
    "Input harga saham bulanan → hitung Volatilitas & VaR."
)

with st.expander("📘 Definisi Variabel & Catatan Metodologi"):
    st.markdown("""
**Variabel yang dihitung**
- **ROA** = Laba Bersih / Total Aset × 100
- **ROE** = Laba Bersih / Total Ekuitas × 100
- **NPM** = Laba Bersih / Pendapatan Operasional × 100
- **DER** = Total Utang / Total Ekuitas × 100
- **NPL** = Kredit Bermasalah / Total Kredit × 100
- **EPS** = Laba Bersih (Rp) / Jumlah Saham Beredar

**Risiko pasar**
- **Volatilitas** dihitung dari **standar deviasi return bulanan** lalu diannualisasi dengan **√12**
- **VaR** pada aplikasi ini adalah **Parametric VaR return bulanan 95% (%)**, sehingga dibaca sebagai **potensi kerugian return dalam persen**, **bukan VaR nominal rupiah**

**Catatan**
- Semua nilai laporan keuangan diinput dalam **juta rupiah**
- **Jumlah saham beredar** diinput dalam **lembar**
- Harga saham diinput dalam **rupiah per lembar**
- Untuk Volatilitas & VaR, disarankan memasukkan **12 harga bulanan**
    """)

st.markdown("---")

# =========================================================
# INPUT TICKER
# =========================================================
col_t1, col_t2 = st.columns([2, 3])
with col_t1:
    # FIX #1: tambah key= pada widget ticker
    ticker_input = st.text_input(
        "Kode Saham (Ticker)",
        value=st.session_state["ticker_name"],
        placeholder="contoh: BBRI",
        key="input_ticker"
    )
    if ticker_input:
        new_ticker = ticker_input.upper().strip()
        # FIX #3: warning jika ticker berubah sementara data periode masih ada
        if (
            st.session_state["ticker_name"]
            and new_ticker != st.session_state["ticker_name"]
            and st.session_state["periods"]
        ):
            st.warning(
                f"⚠️ Ticker diubah dari **{st.session_state['ticker_name']}** ke **{new_ticker}** "
                f"sementara masih ada {len(st.session_state['periods'])} data periode. "
                "Klik **Reset Semua Periode** terlebih dahulu jika ingin menginput saham baru."
            )
        st.session_state["ticker_name"] = new_ticker

# =========================================================
# FORM INPUT PERIODE
# =========================================================
st.markdown("### 📄 Input Laporan Keuangan per Periode")
st.caption(
    "Isi angka dalam satuan **juta rupiah** "
    "(kecuali Jumlah Saham Beredar dalam lembar, dan harga saham dalam rupiah)."
)

# FIX #2: gunakan counter sebagai suffix key agar reset benar-benar mengosongkan form
_k = st.session_state["form_reset_counter"]

with st.expander("➕ Tambah / Update Periode", expanded=True):
    st.markdown("**Data Laporan Laba Rugi & Neraca**")

    c1, c2, c3 = st.columns(3)

    with c1:
        # FIX #1: semua widget punya key= unik
        inp_tahun = st.number_input("Tahun", min_value=2000, max_value=2035, value=2023, step=1, key=f"inp_tahun_{_k}")
        inp_laba = st.number_input("Laba Bersih (juta Rp)", value=0.0, step=100.0, format="%.2f", key=f"inp_laba_{_k}")
        inp_pendapatan = st.number_input("Pendapatan Operasional (juta Rp)", value=0.0, step=100.0, format="%.2f", key=f"inp_pendapatan_{_k}")

    with c2:
        inp_aset = st.number_input("Total Aset (juta Rp)", value=0.0, step=100.0, format="%.2f", key=f"inp_aset_{_k}")
        inp_ekuitas = st.number_input("Total Ekuitas (juta Rp)", value=0.0, step=100.0, format="%.2f", key=f"inp_ekuitas_{_k}")
        inp_utang = st.number_input("Total Utang (juta Rp)", value=0.0, step=100.0, format="%.2f", key=f"inp_utang_{_k}")

    with c3:
        inp_kredit_bermasalah = st.number_input("Kredit Bermasalah / NPL Gross (juta Rp)", value=0.0, step=100.0, format="%.2f", key=f"inp_npl_{_k}")
        inp_total_kredit = st.number_input("Total Kredit yang Disalurkan (juta Rp)", value=0.0, step=100.0, format="%.2f", key=f"inp_kredit_{_k}")
        inp_saham_beredar = st.number_input("Jumlah Saham Beredar (lembar)", value=0.0, step=1000000.0, format="%.0f", key=f"inp_saham_{_k}")

    st.markdown("**Data Harga Saham (untuk Volatilitas & VaR)**")
    st.caption("Masukkan harga saham penutupan bulanan selama periode tahun tersebut, pisahkan dengan koma. Disarankan 12 data.")
    inp_harga_raw = st.text_input(
        "Harga Saham Bulanan (Rp), pisah koma",
        placeholder="contoh: 4500, 4600, 4450, 4700, 4800, 4650, 4900, 5000, 4850, 4950, 5100, 5200",
        key=f"inp_harga_{_k}"
    )

    col_btn1, col_btn2 = st.columns([2, 1])

    with col_btn1:
        btn_hitung = st.button("Hitung & Tambahkan ke Daftar", use_container_width=True, key=f"btn_hitung_{_k}")

    with col_btn2:
        # FIX #2: tombol kosongkan form benar-benar reset dengan menaikkan counter
        btn_kosong = st.button("Kosongkan Form", use_container_width=True, key=f"btn_kosong_{_k}")

    # Proses tombol Hitung
    if btn_hitung:
        error_flag = False

        if not st.session_state["ticker_name"]:
            st.warning("Silakan isi kode saham (ticker) terlebih dahulu.")
            error_flag = True

        if not error_flag:
            try:
                harga_list = parse_harga_list(inp_harga_raw)
            except ValueError as e:
                st.error(str(e))
                error_flag = True

        if not error_flag:
            if 0 < len(harga_list) < 12:
                st.warning(
                    f"Anda hanya memasukkan {len(harga_list)} harga. "
                    "Volatilitas/VaR tetap dihitung, tetapi hasil akan lebih stabil jika menggunakan 12 harga bulanan."
                )

            hasil = hitung_variabel(
                laba=inp_laba,
                pendapatan=inp_pendapatan,
                aset=inp_aset,
                ekuitas=inp_ekuitas,
                utang=inp_utang,
                npl_gross=inp_kredit_bermasalah,
                total_kredit=inp_total_kredit,
                saham_beredar=inp_saham_beredar,
                harga_list=harga_list
            )

            periode_baru = {
                "Ticker": st.session_state["ticker_name"],
                "Tahun": int(inp_tahun),
                "ROA": hasil["ROA"],
                "ROE": hasil["ROE"],
                "NPM": hasil["NPM"],
                "DER": hasil["DER"],
                "NPL": hasil["NPL"],
                "Volatilitas": hasil["Volatilitas"],
                "VaR": hasil["VaR"],
                "EPS": hasil["EPS"],
                "Jumlah Data Harga": hasil["Jumlah_Data_Harga"]
            }

            tahun_list = [p["Tahun"] for p in st.session_state["periods"]]
            if int(inp_tahun) in tahun_list:
                idx = tahun_list.index(int(inp_tahun))
                st.session_state["periods"][idx] = periode_baru
                st.success(f"Data tahun {int(inp_tahun)} berhasil diperbarui.")
            else:
                st.session_state["periods"].append(periode_baru)
                st.success(f"Periode {int(inp_tahun)} berhasil ditambahkan.")

            st.rerun()

    # Proses tombol Kosongkan Form
    if btn_kosong:
        st.session_state["form_reset_counter"] += 1
        st.rerun()

# =========================================================
# HASIL PER PERIODE
# =========================================================
if st.session_state["periods"]:
    st.markdown("---")
    st.markdown("### 📊 Hasil Perhitungan per Periode")

    df_periods = periods_to_dataframe(st.session_state["periods"])

    with st.expander("📋 Tabel Hasil Semua Periode", expanded=False):
        st.dataframe(df_periods, use_container_width=True)

    for _, row in df_periods.iterrows():
        rows_html = ""
        for f in FEATURES:
            rows_html += (
                f"<div class='result-row'>"
                f"<span class='result-label'>{f}</span>"
                f"<span class='result-value'>{fmt_num(row[f], 4)} {SATUAN[f]}</span>"
                f"</div>"
            )

        note_harga = f"<div class='small-note'>Jumlah data harga: {int(row['Jumlah Data Harga'])}</div>"

        st.markdown(f"""
        <div class="section-box">
            <span class='pill-year'>{int(row['Tahun'])}</span>
            <b>{row['Ticker']}</b>
            {note_harga}
            <div class="result-box">{rows_html}</div>
        </div>
        """, unsafe_allow_html=True)

    avg = None
    if len(df_periods) > 1:
        st.markdown("### 📈 Rata-rata Semua Periode")
        avg = df_periods[FEATURES].mean()

        avg_html = ""
        for f in FEATURES:
            avg_html += (
                f"<div class='result-row'>"
                f"<span class='result-label'>{f} (rata-rata {len(df_periods)} periode)</span>"
                f"<span class='result-value'>{fmt_num(avg[f], 4)} {SATUAN[f]}</span>"
                f"</div>"
            )

        st.markdown(
            f'<div class="avg-box"><div class="result-box">{avg_html}</div></div>',
            unsafe_allow_html=True
        )

    # =====================================================
    # PILIH DATA FINAL UNTUK LIST FCM
    # =====================================================
    st.markdown("---")
    st.markdown("### ➕ Tambahkan ke List Input FCM")

    ticker_final = st.session_state["ticker_name"] or "SAHAM"

    if len(df_periods) > 1:
        mode_input = st.radio(
            "Masukkan ke list input FCM sebagai:",
            options=["Rata-rata semua periode", "Pilih tahun tertentu"],
            horizontal=True,
            key="radio_mode_fcm"
        )

        if mode_input == "Pilih tahun tertentu":
            tahun_pilihan = st.selectbox("Pilih tahun", df_periods["Tahun"].tolist(), key="sel_tahun_fcm")
            row_selected = df_periods[df_periods["Tahun"] == tahun_pilihan].iloc[0]
            final_values = {f: row_selected[f] for f in FEATURES}
        else:
            final_values = {f: round(avg[f], 4) for f in FEATURES}
    else:
        final_values = {f: df_periods.iloc[0][f] for f in FEATURES}

    col_add, col_reset = st.columns([2, 1])

    with col_add:
        if st.button(f"✅ Tambahkan {ticker_final} ke List Input FCM", use_container_width=True, key="btn_add_fcm"):
            entry = {"Ticker": ticker_final, **final_values}

            existing_tickers = [x["Ticker"] for x in st.session_state["fcm_input_list"]]
            if ticker_final in existing_tickers:
                idx = existing_tickers.index(ticker_final)
                st.session_state["fcm_input_list"][idx] = entry
                st.success(f"{ticker_final} diperbarui di list input FCM.")
            else:
                st.session_state["fcm_input_list"].append(entry)
                st.success(f"{ticker_final} berhasil ditambahkan ke list input FCM.")

    with col_reset:
        if st.button("🗑️ Reset Semua Periode", use_container_width=True, key="btn_reset_periods"):
            st.session_state["periods"] = []
            st.session_state["ticker_name"] = ""
            st.session_state["form_reset_counter"] += 1
            st.rerun()

    # =====================================================
    # HAPUS PERIODE TERTENTU
    # =====================================================
    st.markdown("### 🗑️ Hapus Periode Tertentu")
    col_del1, col_del2 = st.columns([2, 1])

    with col_del1:
        tahun_hapus = st.selectbox(
            "Pilih tahun yang ingin dihapus",
            df_periods["Tahun"].tolist(),
            key="tahun_hapus"
        )

    with col_del2:
        if st.button("Hapus Tahun Terpilih", use_container_width=True, key="btn_hapus_tahun"):
            st.session_state["periods"] = [
                p for p in st.session_state["periods"] if p["Tahun"] != tahun_hapus
            ]
            st.success(f"Periode tahun {tahun_hapus} berhasil dihapus.")
            st.rerun()

    # =====================================================
    # DOWNLOAD HASIL PERIODE
    # =====================================================
    st.markdown("### ⬇️ Download Hasil Perhitungan Periode")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        csv_periods = df_periods.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Hasil Periode",
            csv_periods,
            file_name=f"hasil_periode_{ticker_final}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_csv_periods"
        )

    with col_d2:
        excel_periods = create_excel_file(df_periods=df_periods)
        st.download_button(
            "Download Excel Hasil Periode",
            excel_periods,
            file_name=f"hasil_periode_{ticker_final}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_excel_periods"
        )

# =========================================================
# LIST INPUT FCM
# =========================================================
if st.session_state["fcm_input_list"]:
    st.markdown("---")
    st.markdown("### 📋 List Input FCM (Siap Dipakai)")

    df_fcm = pd.DataFrame(st.session_state["fcm_input_list"])
    st.dataframe(df_fcm, use_container_width=True)

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

    with col_f1:
        csv_fcm = df_fcm.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV untuk FCM",
            csv_fcm,
            "input_fcm.csv",
            "text/csv",
            use_container_width=True,
            key="dl_csv_fcm"
        )

    with col_f2:
        excel_fcm = create_excel_file(df_fcm=df_fcm)
        st.download_button(
            "⬇️ Download Excel untuk FCM",
            excel_fcm,
            "input_fcm.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_excel_fcm"
        )

    with col_f3:
        if st.button("🗑️ Kosongkan List FCM", use_container_width=True, key="btn_kosong_fcm"):
            st.session_state["fcm_input_list"] = []
            st.rerun()

# =========================================================
# DATA DUMMY
# =========================================================
st.markdown("---")
with st.expander("📌 Data Dummy — Coba Inputkan Ini"):
    st.markdown("""
### **Ticker: BBRI — Tahun 2022**

| Field | Nilai |
|---|---|
| Laba Bersih | 39.279.000 juta Rp |
| Pendapatan Operasional | 109.326.000 juta Rp |
| Total Aset | 1.865.267.000 juta Rp |
| Total Ekuitas | 240.529.000 juta Rp |
| Total Utang | 1.624.738.000 juta Rp |
| Kredit Bermasalah | 23.145.000 juta Rp |
| Total Kredit | 1.040.227.000 juta Rp |
| Jumlah Saham Beredar | 150093601300 lembar |
| Harga Saham Bulanan | `4350, 4390, 4280, 4500, 4620, 4550, 4700, 4810, 4760, 4900, 5050, 5175` |

### **Hasil yang diharapkan (approx)**
- **ROA** ≈ 2.11%
- **ROE** ≈ 16.33%
- **NPM** ≈ 35.93%
- **DER** ≈ 675.4%
- **NPL** ≈ 2.22%
- **EPS** ≈ 261–262 Rp
- **Volatilitas** ≈ ~8% (annualized)
- **VaR** ≈ kisaran beberapa persen
    """)
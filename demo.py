# ============================================================================
# IMPORT LIBRARY
# ============================================================================
import streamlit as st  # Framework untuk membangun web application interaktif
import pandas as pd  # Library untuk manipulasi dan analisis data tabular
import numpy as np  # Library untuk operasi numerik dan array multidimensi
import skfuzzy as fuzz  # Library untuk Fuzzy C-Means clustering algorithm
from sklearn.preprocessing import MinMaxScaler  # Library untuk normalisasi data ke rentang 0-1
import matplotlib.pyplot as plt  # Library untuk visualisasi data dan plotting
import seaborn as sns  # Library untuk visualisasi statistik yang lebih informatif
from io import BytesIO  # Library untuk menangani operasi file dalam memory

# ============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="FCM Saham Perbankan",  # Judul yang muncul di tab browser
    page_icon="📊",  # Icon yang muncul di tab browser
    layout="wide"  # Menggunakan layout lebar untuk tampilan optimal
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #1f2937;
}
[data-testid="stMetric"] {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# KONSTANTA DAN PARAMETER GLOBAL
# ============================================================================
FEATURES = [
    'ROA',  # Return on Asset - Mengukur profitabilitas perusahaan
    'ROE',  # Return on Equity - Mengukur efisiensi penggunaan modal
    'NPM',  # Net Profit Margin - Mengukur margin keuntungan bersih
    'DER',  # Debt to Equity Ratio - Mengukur tingkat leverage keuangan
    'NPL',  # Non Performing Loan - Mengukur risiko kredit bermasalah
    'Volatilitas',  # Mengukur fluktuasi harga saham
    'VaR',  # Value at Risk - Mengukur risiko kerugian maksimal
    'EPS'  # Earnings per Share - Mengukur laba per saham
]

# Parameter algoritma Fuzzy C-Means
n_clusters = 3  # Jumlah cluster yang akan dibentuk (Low, Medium, High Risk)
m_fuzzy = 2.0  # Fuzziness exponent (semakin tinggi nilai, semakin fuzzy)
max_iter = 1000  # Maksimum iterasi untuk mencapai konvergensi
error_tol = 0.005  # Toleransi error untuk menghentikan iterasi
RANDOM_SEED = 42  # Fixed seed untuk menjamin reproduksibilitas hasil

# ============================================================================
# HEADER DAN JUDUL APLIKASI
# ============================================================================
st.title("📊 Klasterisasi Risiko & Kinerja Saham Perbankan")
st.caption("Metode Fuzzy C-Means (FCM) - Hasil Konsisten dengan Fixed Seed")

# ============================================================================
# SIDEBAR - MENAMPILKAN PARAMETER
# ============================================================================
with st.sidebar:
    st.subheader("Parameter")
    st.metric("Jumlah Cluster", n_clusters)
    st.metric("Fuzziness Exponent (m)", m_fuzzy)
    st.metric("Max Iteration", max_iter)
    st.metric("Error Tolerance", error_tol)
    st.metric("Random Seed", RANDOM_SEED)
    st.caption("🔒 Fixed seed = 42 menjamin hasil konsisten setiap running")

# ============================================================================
# TAB NAVIGASI - MEMBAGI ALUR KERJA MENJADI 4 TAB
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Input Data",
    "Preprocessing",
    "Modeling",
    "Hasil"
])

# ============================================================================
# TAB 1: INPUT DATA
# Fungsi: Mengupload file Excel, menampilkan template, dan menyediakan data editor
# ============================================================================
with tab1:
    st.subheader("📥 Import Data dari Excel")
    
    # --- MEMBUAT TEMPLATE EXCEL UNTUK DOWNLOAD ---
    template_df = pd.DataFrame({
        'Ticker': [''],
        'ROA': [0],
        'ROE': [0],
        'NPM': [0],
        'DER': [0],
        'NPL': [0],
        'Volatilitas': [0],
        'VaR': [0],
        'EPS': [0]
    })
    
    # Menyimpan template ke dalam memory sebagai file Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, sheet_name='Template Data', index=False)
    excel_data = output.getvalue()
    
    # --- LAYOUT KOLOM UNTUK DOWNLOAD DAN UPLOAD ---
    col1, col2 = st.columns([2, 1])
    with col1:
        # Tombol untuk mendownload template Excel
        st.download_button(
            label="📎 Download Template Excel",
            data=excel_data,
            file_name="template_fcm_saham.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # Menampilkan instruksi penggunaan
        st.markdown("---")
        st.markdown("**Instruksi:**")
        st.markdown("""
        1. Download template Excel di atas
        2. Isi data saham (jangan ubah nama kolom)
        3. Upload file Excel
        4. Data siap diproses
        """)
        
        # Komponen untuk upload file Excel
        uploaded_file = st.file_uploader(
            "Upload file Excel",
            type=['xlsx', 'xls'],
            key="excel_uploader"
        )
    
    # --- INISIALISASI SESSION STATE UNTUK MENYIMPAN DATA ---
    if 'df_main' not in st.session_state:
        st.session_state['df_main'] = None
    
    # --- PROSES UPLOAD FILE EXCEL ---
    if uploaded_file is not None:
        try:
            # Membaca file Excel yang diupload
            df_import = pd.read_excel(uploaded_file, header=0)
            
            # Validasi: Memastikan semua kolom yang diperlukan ada
            required_columns = ['Ticker'] + FEATURES
            missing_columns = [col for col in required_columns if col not in df_import.columns]
            
            if missing_columns:
                st.error(f"❌ Kolom hilang: {', '.join(missing_columns)}")
            else:
                # --- PEMBERSIHAN DATA ---
                # Menghapus baris yang semua nilainya kosong
                df_import = df_import.dropna(how='all')
                # Menghapus baris dengan Ticker kosong
                df_import = df_import[df_import['Ticker'].notna()]
                # Menghapus baris yang mengandung kata 'Ticker' (header duplikat)
                df_import = df_import[~df_import['Ticker'].astype(str).str.contains('Ticker', case=False, na=False)]
                
                # Mengkonversi kolom fitur ke tipe numerik (non-numerik menjadi NaN)
                for col in FEATURES:
                    df_import[col] = pd.to_numeric(df_import[col], errors='coerce')
                
                # Menghapus baris yang semua fiturnya NaN (tidak ada data sama sekali)
                df_import = df_import.dropna(subset=FEATURES, how='all')
                
                # Validasi akhir: Pastikan ada data yang valid
                if len(df_import) == 0:
                    st.error("❌ Tidak ada data valid")
                else:
                    st.success(f"✅ Berhasil mengimport {len(df_import)} data saham!")
                    st.session_state['df_main'] = df_import.copy()
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # --- DATA EDITOR UNTUK EDIT DATA LANGSUNG DI APLIKASI ---
    if st.session_state['df_main'] is not None:
        st.markdown("---")
        st.subheader("✏️ Edit Data (Opsional)")
        
        # Menampilkan data editor interaktif
        df_edit = st.data_editor(
            st.session_state['df_main'],
            use_container_width=True,
            num_rows="dynamic",  # Memungkinkan penambahan/pengurangan baris
            key="data_editor_main"
        )
        
        # Update session state jika ada perubahan data
        if not df_edit.equals(st.session_state['df_main']):
            st.session_state['df_main'] = df_edit.copy()
        
        # Tombol untuk mereset semua data
        if st.button("🗑️ Reset Semua Data"):
            st.session_state['df_main'] = None
            st.rerun()
        
        # --- MENAMPILKAN STATISTIK DESKRIPTIF ---
        # Hanya ditampilkan jika jumlah data mencukupi (minimal n_clusters + 2)
        if len(df_edit) >= n_clusters + 2:
            st.subheader("📊 Statistik Deskriptif")
            st.dataframe(df_edit[FEATURES].describe().round(4), use_container_width=True)
            
            # Visualisasi korelasi antar variabel menggunakan heatmap
            st.subheader("📈 Korelasi Variabel")
            fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
            sns.heatmap(df_edit[FEATURES].astype(float).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.warning(f"⚠️ Minimal {n_clusters + 2} saham")
    
    else:
        st.info("📌 Silakan upload file Excel")

# ============================================================================
# TAB 2: PREPROCESSING
# Fungsi: Menangani missing value dan melakukan normalisasi data
# ============================================================================
with tab2:
    # --- VALIDASI KETERSEDIAAN DATA ---
    if st.session_state.get('df_main') is None:
        st.warning("⚠️ Belum ada data. Upload file Excel terlebih dahulu.")
        st.stop()
    
    # Copy data untuk preprocessing (menghindari modifikasi data asli)
    df_prep = st.session_state['df_main'].copy()
    
    # Validasi jumlah data minimal
    if len(df_prep) < n_clusters + 2:
        st.error(f"❌ Data minimal {n_clusters + 2} saham")
        st.stop()
    
    # --- HANDLING MISSING VALUE ---
    # Menghitung total missing value pada semua fitur
    missing = df_prep[FEATURES].isnull().sum().sum()
    
    # Menampilkan metrik data
    col1, col2 = st.columns(2)
    col1.metric("Jumlah Data", len(df_prep))
    col2.metric("Missing Value", int(missing))
    
    # Imputasi missing value dengan nilai median (lebih robust terhadap outlier)
    if missing > 0:
        st.warning(f"⚠️ Ditemukan {missing} nilai kosong, diisi dengan median")
        df_prep[FEATURES] = df_prep[FEATURES].fillna(df_prep[FEATURES].median())
    
    # --- NORMALISASI DATA DENGAN MINMAX SCALER ---
    # Mengubah data ke rentang 0-1 untuk menghindari dominasi variabel dengan skala besar
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_prep[FEATURES])
    
    # Membuat DataFrame hasil normalisasi
    df_norm = pd.DataFrame(data_scaled, columns=FEATURES)
    df_norm.insert(0, 'Ticker', df_prep['Ticker'].values)
    
    st.subheader("Data Normalisasi (0-1)")
    st.dataframe(df_norm.round(4), use_container_width=True)
    
    # Menyimpan hasil preprocessing ke session state
    st.session_state['data_scaled'] = data_scaled
    st.session_state['df_prep'] = df_prep

# ============================================================================
# TAB 3: MODELING
# Fungsi: Menjalankan algoritma Fuzzy C-Means clustering
# ============================================================================
with tab3:
    # --- VALIDASI DATA TERSEDIA ---
    if 'data_scaled' not in st.session_state:
        st.warning("⚠️ Belum ada data. Upload file Excel terlebih dahulu.")
        st.stop()
    
    # Mengambil data dari session state
    data_scaled = st.session_state['data_scaled']
    df_prep = st.session_state['df_prep']
    data_T = data_scaled.T  # Transpose untuk FCM (sampel x fitur -> fitur x sampel)
    
    # --- MENJALANKAN FUZZY C-MEANS ---
    with st.spinner("Sedang melakukan clustering dengan FCM..."):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_T,  # Data yang sudah ditranspose
            c=n_clusters,  # Jumlah cluster yang diinginkan
            m=m_fuzzy,  # Fuzziness exponent
            error=error_tol,  # Toleransi error untuk konvergensi
            maxiter=max_iter,  # Maksimum iterasi
            seed=RANDOM_SEED  # Fixed seed untuk konsistensi hasil
        )
    
    # Mengambil label cluster dari matriks keanggotaan (nilai tertinggi)
    cluster_labels = np.argmax(u, axis=0)
    
    # --- MENAMPILKAN METRIK HASIL CLUSTERING ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Cluster", n_clusters)
    col2.metric("FPC", round(fpc, 4))  # Fuzzy Partition Coefficient (0-1, semakin tinggi semakin baik)
    col3.metric("Iterasi", p)  # Jumlah iterasi yang dilakukan
    col4.metric("Random Seed", RANDOM_SEED)
    
    # Interpretasi kualitas clustering berdasarkan nilai FPC
    if fpc >= 0.7:
        st.success(f"✅ Kualitas Cluster SANGAT BAIK (FPC = {round(fpc, 4)})")
    elif fpc >= 0.6:
        st.info(f"📊 Kualitas Cluster BAIK (FPC = {round(fpc, 4)})")
    else:
        st.info(f"📊 Kualitas Cluster CUKUP (FPC = {round(fpc, 4)}) - Karakteristik data overlap")
    
    st.caption(f"🔒 Fixed seed = {RANDOM_SEED} → hasil KONSISTEN setiap running")
    
    # --- MEMBUAT DATAFRAME DERAJAT KEANGGOTAAN ---
    membership = pd.DataFrame(u.T, columns=[f"Cluster {i+1}" for i in range(n_clusters)])
    membership.insert(0, 'Ticker', df_prep['Ticker'].values)
    membership['Cluster'] = cluster_labels + 1  # +1 untuk 1-based indexing
    
    st.subheader("Derajat Keanggotaan (Membership Degree)")
    st.dataframe(membership.round(4), use_container_width=True)
    
    # --- VISUALISASI KONVERGENSI FUNGSI OBJEKTIF ---
    # Menampilkan penurunan nilai fungsi objektif selama iterasi
    fig_jm, ax_jm = plt.subplots(figsize=(8, 3))
    ax_jm.plot(jm, linewidth=2)
    ax_jm.set_title(f"Konvergensi Fungsi Objektif FCM (Seed: {RANDOM_SEED})")
    ax_jm.set_xlabel("Iterasi")
    ax_jm.set_ylabel("Nilai Fungsi Objektif")
    ax_jm.grid(True, alpha=0.3)
    st.pyplot(fig_jm)
    
    # Menyimpan hasil ke session state untuk digunakan di tab Hasil
    st.session_state['cluster_labels'] = cluster_labels
    st.session_state['membership'] = membership
    st.session_state['fpc'] = fpc

# ============================================================================
# TAB 4: HASIL
# Fungsi: Menampilkan dan menginterpretasikan hasil clustering
# ============================================================================
with tab4:
    # --- VALIDASI HASIL CLUSTERING ---
    if 'cluster_labels' not in st.session_state:
        st.warning("⚠️ Belum ada hasil clustering.")
        st.stop()
    
    # Mengambil data dan hasil clustering
    df_prep = st.session_state['df_prep'].copy()
    cluster_labels = st.session_state['cluster_labels']
    df_prep['Cluster'] = cluster_labels
    
    # --- MEMBUAT LABEL PROFIL BERDASARKAN TINGKAT RISIKO ---
    # Menghitung rata-rata risiko (NPL, Volatilitas, VaR) per cluster
    cluster_risk = df_prep.groupby('Cluster')[['NPL', 'Volatilitas', 'VaR']].mean().mean(axis=1)
    # Mengurutkan cluster berdasarkan tingkat risiko (dari rendah ke tinggi)
    sorted_clusters = cluster_risk.sort_values().index.tolist()
    
    # Mapping label untuk setiap cluster
    cluster_map = {}
    for i, cluster in enumerate(sorted_clusters):
        if i == 0:  # Cluster dengan risiko terendah
            cluster_map[cluster] = "🟢 Low Risk — High Return"
        elif i == len(sorted_clusters) - 1:  # Cluster dengan risiko tertinggi
            cluster_map[cluster] = "🔴 High Risk — Low Return"
        else:  # Cluster dengan risiko menengah
            cluster_map[cluster] = "🟡 Medium Risk — Medium Return"
    
    df_prep['Profil'] = df_prep['Cluster'].map(cluster_map)
    
    # --- MENAMPILKAN HASIL KLAS TERISASI ---
    st.subheader("📋 Hasil Klasterisasi")
    hasil = df_prep.copy()
    hasil['Cluster'] = hasil['Cluster'] + 1  # Konversi ke 1-based indexing
    st.dataframe(hasil, use_container_width=True)
    
    # --- MENAMPILKAN RATA-RATA PER CLUSTER ---
    st.subheader("📊 Rata-rata per Cluster")
    summary = hasil.groupby('Profil')[FEATURES].mean().round(4)
    st.dataframe(summary, use_container_width=True)
    
    # --- VISUALISASI SCATTER PLOT (ROA vs NPL) ---
    # Menampilkan sebaran data berdasarkan cluster dengan warna berbeda
    st.subheader("📈 Visualisasi Cluster (ROA vs NPL)")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    
    # Mapping warna untuk setiap profil risiko
    colors_map = {
        "🟢 Low Risk — High Return": 'green',
        "🟡 Medium Risk — Medium Return": 'orange',
        "🔴 High Risk — Low Return": 'red'
    }
    
    # Plot setiap cluster dengan warna dan label yang sesuai
    for profil, color in colors_map.items():
        mask = df_prep['Profil'] == profil
        if mask.any():
            ax_scatter.scatter(
                df_prep.loc[mask, 'ROA'],
                df_prep.loc[mask, 'NPL'],
                s=150,  # Ukuran titik
                c=color,  # Warna sesuai profil
                label=profil,  # Label untuk legend
                alpha=0.7,  # Transparansi
                edgecolors='black',  # Warna tepi titik
                linewidth=1.5  # Tebal tepi
            )
            # Menambahkan label ticker pada setiap titik
            for _, row in df_prep[mask].iterrows():
                ax_scatter.annotate(row['Ticker'], (row['ROA'], row['NPL']), xytext=(5, 5), textcoords='offset points')
    
    ax_scatter.set_xlabel("ROA (%)")
    ax_scatter.set_ylabel("NPL (%)")
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    st.pyplot(fig_scatter)
    
    # --- HEATMAP DERAJAT KEANGGOTAAN ---
    # Visualisasi matriks keanggotaan setiap data ke setiap cluster
    st.subheader("🔥 Heatmap Derajat Keanggotaan")
    heatmap_df = st.session_state['membership'].set_index('Ticker')
    fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_df.iloc[:, :-1], annot=True, cmap="YlGnBu", fmt=".3f", ax=ax_heat)
    st.pyplot(fig_heat)
    
    # --- TOMBOL DOWNLOAD HASIL ---
    # Mengekspor hasil clustering ke file CSV
    csv = hasil.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "hasil_fcm.csv", "text/csv", use_container_width=True)
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
    page_title="FCM Saham Perbankan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING — SIMPLE & MINIMALIS
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1050px;
}

/* Sembunyikan elemen default streamlit yang tidak perlu */
#MainMenu, footer { visibility: hidden; }

h1, h2, h3, h4 { color: #111827; font-weight: 700; letter-spacing: -0.02em; }
p, li, span { color: #374151; }

:root { --accent: #4f46e5; --accent-light: #eef2ff; }

/* Fade-in halus untuk konten utama */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
section.main .block-container > div { animation: fadeIn 0.3s ease-out; }

/* === Hero / Home === */
.hero {
    background: #111827;
    border-radius: 16px;
    padding: 40px 36px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 { color: white; font-size: 1.9rem; margin-bottom: 8px; }
.hero p { color: #d1d5db; font-size: 1rem; max-width: 600px; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    margin-bottom: 12px;
    color: #e5e7eb;
}

/* === Card === */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    border-color: var(--accent);
    transform: translateY(-3px);
    box-shadow: 0 10px 24px -8px rgba(79, 70, 229, 0.25);
}

.card-step {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: background 0.2s ease, border-left-width 0.2s ease, transform 0.15s ease;
}
.card-step:hover {
    background: var(--accent-light);
    border-left-width: 5px;
    transform: translateX(2px);
}
.card-step b { color: #111827; }

.feature-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.feature-item {
    flex: 1 1 200px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.feature-item:hover {
    border-color: var(--accent);
    transform: translateY(-3px);
    box-shadow: 0 10px 24px -8px rgba(79, 70, 229, 0.25);
}
.feature-item b { display:block; margin: 6px 0 4px 0; color:#111827; font-size: 0.92rem;}
.feature-item span { font-size: 0.82rem; color: #6b7280; }

/* === Metric cards === */
[data-testid="stMetric"] {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 12px;
    border-radius: 10px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] { color: #6b7280; font-size: 0.8rem; }

/* === Buttons — diseragamkan: download = dark solid, button biasa = outline accent === */
.stButton button, .stDownloadButton button {
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid #e5e7eb;
    transition: opacity 0.15s ease, background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.stButton button {
    background: #ffffff;
    color: var(--accent);
    border: 1px solid var(--accent);
}
.stButton button:hover {
    background: var(--accent-light);
    opacity: 1;
}
.stButton button[kind="primary"] {
    background: var(--accent);
    border: none;
    color: white;
}
.stButton button[kind="primary"]:hover { opacity: 0.88; }
.stDownloadButton button {
    background: #111827;
    color: white;
    border: none;
}
.stDownloadButton button:hover { opacity: 0.88; }

/* === Tag pill untuk profil risiko === */
.pill {
    display:inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; white-space: normal;
    line-height: 1.4; max-width: 100%;
}
.pill-low { background:#dcfce7; color:#166534; }
.pill-med { background:#fef3c7; color:#92400e; }
.pill-high { background:#fee2e2; color:#991b1b; }

/* === Grid kartu ringkasan profil (Hasil) — flexbox agar tidak overflow saat label panjang/banyak === */
.profile-grid {
    display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 14px;
}
.profile-card {
    flex: 1 1 200px; max-width: calc(25% - 11px);
    text-align: center; margin-bottom: 0;
}
@media (max-width: 900px) {
    .profile-card { max-width: calc(50% - 7px); }
}

/* Icon inline helper untuk judul section */
.icon-title { display:flex; align-items:center; gap:9px; }
.icon-title svg { width:21px; height:21px; stroke: var(--accent); flex-shrink:0; }
.icon-title h2, .icon-title h3 { margin: 0; }

/* === Tab navigasi utama (pengganti sidebar) === */
div[data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 2px solid #e5e7eb;
    position: relative;
    z-index: 1;
    overflow: visible !important;
    flex-wrap: wrap;
}
button[data-baseweb="tab"] {
    height: auto;
    min-height: 50px;
    padding: 12px 20px;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    font-size: 0.98rem;
    color: #6b7280;
    background: transparent;
    transition: background 0.2s ease, color 0.2s ease;
}
button[data-baseweb="tab"]:hover {
    background: var(--accent-light);
    color: var(--accent);
}
button[data-baseweb="tab"] p {
    font-size: 1rem;
    font-weight: 600;
}
/* Grayscale emoji icon di tab supaya terlihat satu warna & profesional,
   warna hanya muncul lewat background/underline saat hover atau aktif */
button[data-baseweb="tab"] p {
    filter: grayscale(1) contrast(0.7) brightness(0.85);
}
button[data-baseweb="tab"][aria-selected="true"] p {
    filter: grayscale(1) contrast(1.3) brightness(0.6) sepia(1) hue-rotate(220deg) saturate(4);
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--accent-light);
}
div[data-baseweb="tab-highlight"] {
    background-color: var(--accent);
    height: 3px;
}
div[data-baseweb="tab-border"] { display: none; }

/* Progress step indicator — minimalis, gaya breadcrumb */
.step-track { display:flex; align-items:center; gap:6px; margin-bottom: 16px; flex-wrap: wrap; }
.step-chip {
    font-size: 0.78rem; font-weight: 500; color: #9ca3af;
    display:flex; align-items:center; gap:5px;
}
.step-chip .dot {
    width:5px; height:5px; border-radius:50%; background:#d1d5db; flex-shrink:0;
}
.step-chip.active { color: var(--accent); font-weight: 700; }
.step-chip.active .dot { background: var(--accent); }
.step-chip.done { color: #6b7280; }
.step-chip.done .dot { background: #16a34a; }
.step-divider { color: #d1d5db; font-size: 0.7rem; }

/* === Brand / logo header (pengganti sidebar) === */
.brand { display:flex; align-items:center; gap:10px; margin-bottom: 2px; }
.brand .logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    display:flex; align-items:center; justify-content:center;
    flex-shrink: 0;
}
.brand .logo svg { width: 19px; height: 19px; stroke: white; }
.brand .title { font-weight: 700; font-size: 1.05rem; color:#111827; line-height:1.2; }
.brand .subtitle { font-size: 0.78rem; color:#9ca3af; }

hr { border-color: #e5e7eb; margin: 8px 0; }

/* === Sidebar — parameter model === */
section[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.3rem; }
.sidebar-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
    color: #9ca3af; text-transform: uppercase; margin: 16px 0 6px 1px;
}
.param-card {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 10px 12px; margin-bottom: 8px;
    transition: border-color 0.2s ease, transform 0.15s ease;
}
.param-card:hover {
    border-color: var(--accent);
    transform: translateX(2px);
}
.param-card .param-row { display:flex; justify-content:space-between; align-items:baseline; }
.param-card .param-name { font-size: 0.82rem; font-weight: 600; color:#111827; }
.param-card .param-value { font-size: 0.95rem; font-weight: 700; color: var(--accent); }
.param-card .param-desc { font-size: 0.74rem; color:#9ca3af; margin-top: 2px; line-height:1.4; }

/* === Fix: header bawaan streamlit menutupi tab saat sticky === */
header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}
div[data-testid="stToolbar"] { right: 1rem; }
.stApp > header { background: transparent; }
div[data-testid="stAppViewContainer"] > .main { padding-top: 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ICON SET (SVG line-icons, profesional — pengganti emoji)
# ============================================================================
ICONS = {
    "home": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/></svg>',
    "book": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    "upload": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "sliders": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "cpu": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/></svg>',
    "result": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/></svg>',
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 14l3-3 3 3 4-5"/></svg>',
    "check": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "lock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    "circle-low": '<svg viewBox="0 0 24 24" fill="#16a34a" stroke="none"><circle cx="12" cy="12" r="8"/></svg>',
    "circle-med": '<svg viewBox="0 0 24 24" fill="#d97706" stroke="none"><circle cx="12" cy="12" r="8"/></svg>',
    "circle-high": '<svg viewBox="0 0 24 24" fill="#dc2626" stroke="none"><circle cx="12" cy="12" r="8"/></svg>',
    "arrow-right": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
}

def icon(name, size=18, color="currentColor"):
    """Render inline SVG icon."""
    svg = ICONS[name].replace('width="24" height="24"', '')
    return f'<span style="display:inline-flex; width:{size}px; height:{size}px; color:{color}; vertical-align:-4px;">{svg}</span>'

def section_title(name, text, level="h2"):
    st.markdown(f'<div class="icon-title">{icon(name, 24)}<{level}>{text}</{level}></div>', unsafe_allow_html=True)

def param_card(name, value, desc):
    st.markdown(f"""
    <div class="param-card">
        <div class="param-row"><span class="param-name">{name}</span><span class="param-value">{value}</span></div>
        <div class="param-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

WORKFLOW_PAGES = ["Input Data", "Preprocessing", "Modeling", "Hasil"]

def step_track(current_page):
    """Render breadcrumb minimalis untuk 4 tahap kerja."""
    chips = []
    current_idx = WORKFLOW_PAGES.index(current_page)
    for i, p in enumerate(WORKFLOW_PAGES):
        if i < current_idx:
            cls = "done"
        elif i == current_idx:
            cls = "active"
        else:
            cls = ""
        chips.append(f'<div class="step-chip {cls}"><span class="dot"></span>{p}</div>')
        if i < len(WORKFLOW_PAGES) - 1:
            chips.append('<span class="step-divider">›</span>')
    st.markdown(f'<div class="step-track">{"".join(chips)}</div>', unsafe_allow_html=True)

# ============================================================================
# KONSTANTA DAN PARAMETER GLOBAL (TIDAK DIUBAH — LOGIKA TETAP SAMA)
# ============================================================================
FEATURES = [
    'ROA', 'ROE', 'NPM', 'DER', 'NPL', 'Volatilitas', 'VaR', 'EPS'
]

FEATURE_DESC = {
    'ROA': 'Return on Asset — profitabilitas perusahaan',
    'ROE': 'Return on Equity — efisiensi penggunaan modal',
    'NPM': 'Net Profit Margin — margin keuntungan bersih',
    'DER': 'Debt to Equity Ratio — tingkat leverage keuangan',
    'NPL': 'Non Performing Loan — risiko kredit bermasalah',
    'Volatilitas': 'Fluktuasi harga saham',
    'VaR': 'Value at Risk — risiko kerugian maksimal',
    'EPS': 'Earnings per Share — laba per saham',
}

N_CLUSTERS_DEFAULT = 3
m_fuzzy = 2.0
max_iter = 1000
error_tol = 0.005
RANDOM_SEED = 42

# ============================================================================
# SESSION STATE DEFAULT
# ============================================================================
if 'df_main' not in st.session_state:
    st.session_state['df_main'] = None
if 'n_clusters' not in st.session_state:
    st.session_state['n_clusters'] = N_CLUSTERS_DEFAULT

# ============================================================================
# SIDEBAR — PARAMETER MODEL & KETERANGAN
# ============================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="brand">
        <div class="logo">{icon('chart', 18, 'white')}</div>
        <div>
            <div class="title">FCM Saham</div>
            <div class="subtitle">Klasterisasi Risiko Perbankan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>Parameter Model</div>", unsafe_allow_html=True)
    st.session_state['n_clusters'] = st.slider(
        "Jumlah Cluster", min_value=2, max_value=6,
        value=st.session_state['n_clusters'], step=1,
        help="Banyaknya kelompok risiko yang dibentuk. Default 3 (Low/Medium/High). Dibatasi sampai 6 karena dengan 8 variabel & data yang biasanya puluhan saham, cluster lebih banyak dari itu cenderung terlalu granular (1-2 saham per cluster) dan kehilangan makna sebagai kelompok."
    )
    n_clusters = st.session_state['n_clusters']
    st.caption(f"{n_clusters} cluster dibentuk dari data. Cek FPC pembanding di tab Modeling untuk menilai apakah jumlah ini optimal.")
    param_card("Fuzziness (m)", m_fuzzy, "Mengatur tingkat ke-fuzzy-an keanggotaan. Semakin besar nilainya, batas antar cluster semakin kabur/melebur.")
    param_card("Max Iteration", max_iter, "Batas maksimum perulangan yang dilakukan algoritma sebelum berhenti, meski belum konvergen.")
    param_card("Error Tolerance", error_tol, "Ambang batas perubahan minimum antar iterasi. Jika perubahan lebih kecil dari ini, proses dianggap konvergen dan berhenti.")
    param_card("Random Seed", RANDOM_SEED, "Angka kunci untuk inisialisasi acak. Dengan nilai tetap, hasil clustering selalu sama di setiap proses running.")

    st.markdown("<div class='sidebar-label'>Status Data</div>", unsafe_allow_html=True)
    _df_main = st.session_state.get('df_main')
    _n_valid = 0
    if _df_main is not None:
        _n_valid = len(_df_main[_df_main['Ticker'].astype(str).str.strip() != ''])
    if _n_valid > 0:
        st.markdown(
            f"<div style='font-size:0.82rem; color:#166534; display:flex; align-items:center; gap:6px;'>"
            f"{icon('check', 14, '#16a34a')} {_n_valid} saham terisi</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='font-size:0.82rem; color:#9ca3af;'>Belum ada data diisi</div>",
            unsafe_allow_html=True
        )

# ============================================================================
# HEADER — LOGO & JUDUL
# ============================================================================
st.markdown(f"""
<div class="brand">
    <div class="logo">{icon('chart', 18, 'white')}</div>
    <div>
        <div class="title">FCM Saham</div>
        <div class="subtitle">Klasterisasi Risiko & Kinerja Saham Perbankan</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ============================================================================
# TAB NAVIGASI UTAMA
# ============================================================================
tab_home, tab_tutorial, tab_input, tab_prep, tab_model, tab_hasil = st.tabs(
    ["🏠  Home", "📘  Tutorial", "📤  Input Data", "🧮  Preprocessing", "⚙️  Modeling", "📊  Hasil"]
)

# ============================================================================
# TAB: HOME
# ============================================================================
with tab_home:
    st.markdown(f"""
    <div class="hero">
        <div class="badge">Fuzzy C-Means Clustering</div>
        <h1>Klasterisasi Risiko & Kinerja Saham Perbankan</h1>
        <p>Kelompokkan saham perbankan ke dalam profil Low, Medium, dan High Risk
        secara otomatis berdasarkan 8 rasio keuangan & risiko, menggunakan algoritma Fuzzy C-Means
        dengan hasil yang konsisten dan dapat direproduksi.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="feature-grid">
        <div class="feature-item">{icon('upload', 20, '#111827')}<b>Import Mudah</b><span>Upload data via template Excel, atau edit langsung di tabel interaktif.</span></div>
        <div class="feature-item">{icon('sliders', 20, '#111827')}<b>Preprocessing Otomatis</b><span>Missing value diisi median, data dinormalisasi ke skala 0–1.</span></div>
        <div class="feature-item">{icon('cpu', 20, '#111827')}<b>FCM Clustering</b><span>Algoritma fuzzy c-means dengan seed tetap untuk hasil stabil.</span></div>
        <div class="feature-item">{icon('result', 20, '#111827')}<b>Hasil Visual</b><span>Tabel, scatter plot, heatmap keanggotaan, dan label profil risiko.</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown("##### Alur Kerja")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("upload", "Input Data", "Upload Excel atau isi manual"),
        ("sliders", "Preprocessing", "Bersihkan & normalisasi data"),
        ("cpu", "Modeling", "Jalankan algoritma FCM"),
        ("result", "Hasil", "Lihat profil risiko tiap saham"),
    ]
    for col, (ic, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                {icon(ic, 22, '#111827')}
                <div style="margin-top:8px;"><b>{title}</b></div>
                <span style="color:#6b7280; font-size:0.82rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("##### Variabel yang Digunakan")
    fcols = st.columns(4)
    for i, feat in enumerate(FEATURES):
        with fcols[i % 4]:
            st.markdown(f"""
            <div class="card" style="padding:14px;">
                <b style="font-size:0.88rem;">{feat}</b><br>
                <span style="font-size:0.76rem; color:#6b7280;">{FEATURE_DESC[feat]}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.info("Buka tab **Tutorial** untuk panduan lengkap, atau langsung ke tab **Input Data** untuk mulai.")

# ============================================================================
# TAB: TUTORIAL
# ============================================================================
with tab_tutorial:
    section_title("book", "Tutorial Penggunaan")
    st.caption("Ikuti langkah-langkah berikut agar hasil klasterisasi optimal.")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    steps = [
        ("Download Template", "Buka tab Input Data, lalu unduh template Excel. Template berisi kolom Ticker dan 8 variabel (ROA, ROE, NPM, DER, NPL, Volatilitas, VaR, EPS)."),
        ("Isi Data", "Isi setiap baris dengan kode saham dan nilai rasio keuangannya. Jangan mengubah nama kolom pada header."),
        ("Upload File", "Upload kembali file Excel yang sudah diisi. Sistem otomatis memvalidasi kolom, membuang baris kosong, dan mengonversi nilai ke format numerik."),
        ("Edit Data (Opsional)", "Data dapat diedit langsung di tabel interaktif tanpa perlu mengunggah ulang file — termasuk menambah atau menghapus baris."),
        ("Preprocessing", "Nilai kosong otomatis diisi dengan median kolom, lalu seluruh data dinormalisasi ke rentang 0–1 menggunakan MinMaxScaler."),
        ("Modeling", "Algoritma Fuzzy C-Means membentuk 3 cluster. Perhatikan nilai FPC (Fuzzy Partition Coefficient): semakin mendekati 1, semakin baik kualitas pemisahan cluster."),
        ("Lihat Hasil", "Lihat label profil risiko (Low, Medium, High), visualisasi sebaran ROA vs NPL, heatmap derajat keanggotaan, dan unduh hasil dalam format CSV."),
    ]
    for i, (title, desc) in enumerate(steps, start=1):
        st.markdown(f"""
        <div class="card-step"><b>{i}. {title}</b><br>{desc}</div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("##### Pertanyaan Umum")
    with st.expander("Berapa minimal jumlah saham yang dibutuhkan?"):
        st.write(f"Minimal **{n_clusters + 2} saham** agar algoritma FCM dapat membentuk {n_clusters} cluster dengan stabil.")
    with st.expander("Apa itu Fuzzy Partition Coefficient (FPC)?"):
        st.write("FPC mengukur seberapa jelas pemisahan antar cluster. Nilai berkisar 0–1: ≥0.7 sangat baik, 0.6–0.7 baik, di bawah itu menunjukkan data saling tumpang tindih (overlap).")
    with st.expander("Mengapa hasil clustering selalu sama setiap dijalankan?"):
        st.write(f"Karena parameter `random_seed` dikunci di angka **{RANDOM_SEED}**, sehingga inisialisasi algoritma selalu identik dan hasil dapat direproduksi.")
    with st.expander("Bagaimana cara penentuan profil Low/Medium/High Risk?"):
        st.write("Rata-rata nilai NPL, Volatilitas, dan VaR dihitung per cluster, lalu diurutkan dari yang terendah ke tertinggi untuk menentukan label risiko.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.info("Lanjut ke tab **Input Data** untuk mulai mengunggah data saham.")

# ============================================================================
# TAB: INPUT DATA
# ============================================================================
with tab_input:
    step_track("Input Data")
    section_title("upload", "Input Data")
    st.caption(f"Isi data saham langsung di tabel (minimal {n_clusters + 2} baris), atau gunakan import Excel sebagai alternatif.")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # --- Inisialisasi 5 baris kosong sebagai default agar bisa langsung diisi manual ---
    EMPTY_ROWS = 5
    if st.session_state['df_main'] is None:
        st.session_state['df_main'] = pd.DataFrame({
            'Ticker': [''] * EMPTY_ROWS,
            **{col: [0.0] * EMPTY_ROWS for col in FEATURES}
        })
    if 'editor_version' not in st.session_state:
        st.session_state['editor_version'] = 0

    # --- DATA EDITOR — cara utama mengisi data ---
    st.markdown("##### Tabel Data Saham")
    st.caption("Klik sel untuk mengisi. Tambah baris dengan tombol (+) di bagian bawah tabel.")

    df_edit = st.data_editor(
        st.session_state['df_main'],
        use_container_width=True,
        num_rows="dynamic",
        key=f"data_editor_main_{st.session_state['editor_version']}"
    )

    if not df_edit.equals(st.session_state['df_main']):
        st.session_state['df_main'] = df_edit.copy()

    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Reset Data", use_container_width=True):
            st.session_state['df_main'] = pd.DataFrame({
                'Ticker': [''] * EMPTY_ROWS,
                **{col: [0.0] * EMPTY_ROWS for col in FEATURES}
            })
            st.session_state['editor_version'] += 1
            st.session_state.pop('_last_uploaded_name', None)
            st.rerun()

    # --- Hitung baris valid (Ticker terisi) untuk validasi minimal data ---
    df_valid = df_edit[df_edit['Ticker'].astype(str).str.strip() != '']

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # --- IMPORT EXCEL — opsi sekunder, disembunyikan dalam expander ---
    with st.expander("Atau import dari file Excel"):
        template_df = pd.DataFrame({
            'Ticker': [''], 'ROA': [0], 'ROE': [0], 'NPM': [0], 'DER': [0],
            'NPL': [0], 'Volatilitas': [0], 'VaR': [0], 'EPS': [0]
        })
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            template_df.to_excel(writer, sheet_name='Template Data', index=False)
        excel_data = output.getvalue()

        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.markdown("**1. Download template, lalu isi data**")
            st.download_button(
                label="Download Template Excel",
                data=excel_data,
                file_name="template_fcm_saham.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.caption("Jangan ubah nama kolom pada header.")
        with col2:
            st.markdown("**2. Upload kembali file Excel**")
            uploaded_file = st.file_uploader(
                "Upload file Excel", type=['xlsx', 'xls'],
                key="excel_uploader", label_visibility="collapsed"
            )

        # --- PROSES UPLOAD FILE EXCEL (LOGIKA TIDAK DIUBAH, hanya ditambah guard anti-loop) ---
        if uploaded_file is not None and st.session_state.get('_last_uploaded_name') != uploaded_file.name:
            try:
                df_import = pd.read_excel(uploaded_file, header=0)

                required_columns = ['Ticker'] + FEATURES
                missing_columns = [col for col in required_columns if col not in df_import.columns]

                if missing_columns:
                    st.error(f"Kolom hilang: {', '.join(missing_columns)}")
                else:
                    df_import = df_import.dropna(how='all')
                    df_import = df_import[df_import['Ticker'].notna()]
                    df_import = df_import[~df_import['Ticker'].astype(str).str.contains('Ticker', case=False, na=False)]

                    for col in FEATURES:
                        df_import[col] = pd.to_numeric(df_import[col], errors='coerce')

                    df_import = df_import.dropna(subset=FEATURES, how='all')

                    if len(df_import) == 0:
                        st.error("Tidak ada data valid")
                    else:
                        st.success(f"Berhasil mengimport {len(df_import)} data saham")
                        st.session_state['df_main'] = df_import.copy()
                        st.session_state['_last_uploaded_name'] = uploaded_file.name
                        st.session_state['editor_version'] = st.session_state.get('editor_version', 0) + 1
                        st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # --- VALIDASI & PREVIEW STATISTIK ---
    if len(df_valid) >= n_clusters + 2:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        tabA, tabB = st.tabs(["Statistik Deskriptif", "Korelasi Variabel"])
        with tabA:
            st.dataframe(df_valid[FEATURES].describe().round(4), use_container_width=True)
        with tabB:
            fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
            sns.heatmap(df_valid[FEATURES].astype(float).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.success("Data siap. Lanjut ke tab **Preprocessing**.")
    else:
        st.warning(f"Minimal {n_clusters + 2} baris dengan Ticker terisi diperlukan untuk melanjutkan. Saat ini: {len(df_valid)}.")

# ============================================================================
# TAB: PREPROCESSING
# ============================================================================
with tab_prep:
    step_track("Preprocessing")
    section_title("sliders", "Preprocessing")
    st.caption("Penanganan missing value & normalisasi data ke skala 0–1.")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    _df_check = st.session_state.get('df_main')
    _df_check_valid = _df_check[_df_check['Ticker'].astype(str).str.strip() != ''] if _df_check is not None else None

    if _df_check_valid is None or len(_df_check_valid) < n_clusters + 2:
        st.warning(f"Minimal {n_clusters + 2} saham dengan Ticker terisi diperlukan. Lengkapi data di tab Input Data terlebih dahulu.")
    else:
        df_prep = _df_check_valid.copy().reset_index(drop=True)

        missing = df_prep[FEATURES].isnull().sum().sum()

        col1, col2 = st.columns(2)
        col1.metric("Jumlah Data", len(df_prep))
        col2.metric("Missing Value", int(missing))

        if missing > 0:
            st.warning(f"Ditemukan {missing} nilai kosong, diisi dengan median")
            df_prep[FEATURES] = df_prep[FEATURES].fillna(df_prep[FEATURES].median())
        else:
            st.success("Tidak ada nilai kosong pada data.")

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_prep[FEATURES])

        df_norm = pd.DataFrame(data_scaled, columns=FEATURES)
        df_norm.insert(0, 'Ticker', df_prep['Ticker'].values)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("##### Data Setelah Normalisasi (0–1)")
        st.dataframe(df_norm.round(4), use_container_width=True)

        st.session_state['data_scaled'] = data_scaled
        st.session_state['df_prep'] = df_prep

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.success("Normalisasi selesai. Lanjut ke tab **Modeling**.")

# ============================================================================
# TAB: MODELING
# ============================================================================
with tab_model:
    step_track("Modeling")
    section_title("cpu", "Modeling — Fuzzy C-Means")
    st.caption("Menjalankan algoritma clustering dengan parameter tetap untuk hasil konsisten.")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if 'data_scaled' not in st.session_state:
        st.warning("Belum ada data. Selesaikan tahap Preprocessing terlebih dahulu.")
    else:
        data_scaled = st.session_state['data_scaled']
        df_prep = st.session_state['df_prep']
        data_T = data_scaled.T

        with st.spinner("Sedang melakukan clustering dengan FCM..."):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data_T,
                c=n_clusters,
                m=m_fuzzy,
                error=error_tol,
                maxiter=max_iter,
                seed=RANDOM_SEED
            )

        cluster_labels = np.argmax(u, axis=0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Cluster", n_clusters)
        col2.metric("FPC", round(fpc, 4))
        col3.metric("Iterasi", p)
        col4.metric("Random Seed", RANDOM_SEED)

        if fpc >= 0.7:
            st.success(f"Kualitas Cluster Sangat Baik (FPC = {round(fpc, 4)})")
            if st.session_state.get('last_fpc_celebrated') != round(fpc, 4):
                st.balloons()
                st.session_state['last_fpc_celebrated'] = round(fpc, 4)
        elif fpc >= 0.6:
            st.info(f"Kualitas Cluster Baik (FPC = {round(fpc, 4)})")
        else:
            st.info(f"Kualitas Cluster Cukup (FPC = {round(fpc, 4)}) — karakteristik data overlap")

        st.markdown(
            f"<div style='font-size:0.82rem; color:#6b7280; display:flex; align-items:center; gap:6px;'>"
            f"{icon('lock', 13, '#6b7280')} Fixed seed = {RANDOM_SEED} → hasil konsisten setiap running</div>",
            unsafe_allow_html=True
        )

        membership = pd.DataFrame(u.T, columns=[f"Cluster {i+1}" for i in range(n_clusters)])
        membership.insert(0, 'Ticker', df_prep['Ticker'].values)
        membership['Cluster'] = cluster_labels + 1

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        tabM, tabC, tabK = st.tabs(["Derajat Keanggotaan", "Konvergensi Fungsi Objektif", "FPC Pembanding"])
        with tabM:
            st.dataframe(membership.round(4), use_container_width=True)
        with tabC:
            fig_jm, ax_jm = plt.subplots(figsize=(8, 3))
            ax_jm.plot(jm, linewidth=2, color="#111827")
            ax_jm.set_title(f"Konvergensi Fungsi Objektif FCM (Seed: {RANDOM_SEED})")
            ax_jm.set_xlabel("Iterasi")
            ax_jm.set_ylabel("Nilai Fungsi Objektif")
            ax_jm.grid(True, alpha=0.3)
            st.pyplot(fig_jm)
        with tabK:
            st.caption("FPC dihitung ulang untuk beberapa nilai k guna membantu menilai apakah jumlah cluster yang dipilih sudah optimal. Semakin mendekati 1, semakin baik pemisahan cluster.")
            k_range = [k for k in range(2, 7) if k <= len(df_prep) - 1]
            fpc_list = []
            for k_test in k_range:
                try:
                    _, _, _, _, _, _, fpc_test = fuzz.cluster.cmeans(
                        data_T, c=k_test, m=m_fuzzy, error=error_tol,
                        maxiter=max_iter, seed=RANDOM_SEED
                    )
                    fpc_list.append(fpc_test)
                except Exception:
                    fpc_list.append(np.nan)

            fig_k, ax_k = plt.subplots(figsize=(8, 3))
            bar_colors = ['#4f46e5' if k_test == n_clusters else '#d1d5db' for k_test in k_range]
            ax_k.bar([str(k_test) for k_test in k_range], fpc_list, color=bar_colors)
            ax_k.set_xlabel("Jumlah Cluster (k)")
            ax_k.set_ylabel("FPC")
            ax_k.set_ylim(0, 1)
            ax_k.axhline(0.7, color='#16a34a', linestyle='--', linewidth=1, alpha=0.6)
            ax_k.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig_k)

            best_k = k_range[int(np.nanargmax(fpc_list))]
            if best_k != n_clusters:
                st.info(f"FPC tertinggi ada di k = {best_k}. Cluster yang sedang dipakai (k = {n_clusters}) bisa disesuaikan di sidebar bila ingin pemisahan yang lebih optimal.")
            else:
                st.success(f"k = {n_clusters} yang sedang dipakai sudah merupakan FPC tertinggi pada rentang k yang diuji.")

        st.session_state['cluster_labels'] = cluster_labels
        st.session_state['membership'] = membership
        st.session_state['fpc'] = fpc

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.success("Clustering selesai. Lanjut ke tab **Hasil**.")

# ============================================================================
# TAB: HASIL
# ============================================================================
with tab_hasil:
    step_track("Hasil")
    section_title("result", "Hasil Klasterisasi")
    st.caption("Profil risiko & kinerja setiap saham berdasarkan hasil Fuzzy C-Means.")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if 'cluster_labels' not in st.session_state:
        st.warning("Belum ada hasil clustering. Selesaikan tahap Modeling terlebih dahulu.")
    else:
        df_prep = st.session_state['df_prep'].copy()
        cluster_labels = st.session_state['cluster_labels']
        df_prep['Cluster'] = cluster_labels
        k = n_clusters

        # --- Skor Risk & Return dihitung terpisah per cluster dari data TERNORMALISASI (0-1), ---
        # --- bukan rata-rata mentah, supaya tidak didominasi variabel berskala besar. ---
        risk_cols = ['NPL', 'Volatilitas', 'VaR']
        return_cols = ['ROA', 'ROE', 'EPS']
        data_scaled_df = pd.DataFrame(st.session_state['data_scaled'], columns=FEATURES)
        data_scaled_df['Cluster'] = cluster_labels
        cluster_risk = data_scaled_df.groupby('Cluster')[risk_cols].mean().mean(axis=1)
        cluster_return = data_scaled_df.groupby('Cluster')[return_cols].mean().mean(axis=1)

        # --- Tiap cluster diberi level Risk & Return SENDIRI-SENDIRI berdasarkan rank-nya, ---
        # --- supaya kombinasi seperti "High Risk — High Return" tetap bisa muncul apa adanya. ---
        def rank_levels(series, k_levels):
            """Bagi cluster ke level Low/Medium/High (atau Low/High untuk 2 cluster) berdasarkan rank nilai."""
            order = series.sort_values().index.tolist()
            n = len(order)
            if n <= 2:
                names = ["Low", "High"]
            elif n == 3:
                names = ["Low", "Medium", "High"]
            else:
                names = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
                idxs = np.linspace(0, len(names) - 1, n).round().astype(int)
                names = [names[i] for i in idxs]
            return {cluster: names[i] for i, cluster in enumerate(order)}

        risk_level_map = rank_levels(cluster_risk, k)
        return_level_map = rank_levels(cluster_return, k)

        cluster_map = {
            c: f"{risk_level_map[c]} Risk — {return_level_map[c]} Return"
            for c in cluster_risk.index
        }
        df_prep['Profil'] = df_prep['Cluster'].map(cluster_map)
        level_names = [cluster_map[c] for c in cluster_risk.sort_values().index.tolist()]

        def pill(label_text, css_class):
            return f'<span class="pill {css_class}">{label_text}</span>'

        # Kelas CSS pill ditentukan dari level Risk cluster tersebut (bukan urutan gabungan)
        def pill_class_for_cluster(cluster_id):
            lvl = risk_level_map[cluster_id]
            if lvl == "Low":
                return 'pill-low'
            elif lvl == "High":
                return 'pill-high'
            return 'pill-med'

        # Ringkasan jumlah saham per profil — pakai flexbox grid HTML agar tidak overflow
        counts = df_prep['Profil'].value_counts()
        unique_profiles = list(dict.fromkeys(level_names))  # urut & tanpa duplikat
        cluster_by_profile = {v: c for c, v in cluster_map.items()}

        cards_html = '<div class="profile-grid">'
        for label in unique_profiles:
            css_cls = pill_class_for_cluster(cluster_by_profile[label])
            cards_html += (
                f"<div class='card profile-card'>{pill(label, css_cls)}"
                f"<h3 style='margin:8px 0 0 0;'>{int(counts.get(label, 0))}</h3></div>"
            )
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Tabel Hasil", "Rata-rata per Cluster", "Scatter Plot", "Heatmap Keanggotaan"])

        hasil = df_prep.copy()
        hasil['Cluster'] = hasil['Cluster'] + 1

        with tab1:
            sort_cols = st.columns([2, 1.6, 2.4])
            with sort_cols[0]:
                sort_by = st.selectbox(
                    "Urutkan berdasarkan",
                    options=['Ticker', 'Profil', 'Cluster'] + FEATURES,
                    index=0,
                    key="sort_by_hasil"
                )
            with sort_cols[1]:
                sort_dir = st.selectbox(
                    "Arah",
                    options=["Tertinggi → Terendah", "Terendah → Tertinggi"],
                    index=0,
                    key="sort_dir_hasil"
                )

            hasil_sorted = hasil.sort_values(
                by=sort_by,
                ascending=(sort_dir == "Terendah → Tertinggi")
            ).reset_index(drop=True)

            # --- Pewarnaan baris berdasar level Risk (bukan return) supaya konsisten dengan pill di atas ---
            row_risk_level = hasil_sorted['Cluster'].apply(lambda c: risk_level_map.get(c - 1, "Medium"))
            row_colors = row_risk_level.map({
                "Low": "background-color: #f0fdf4;",
                "High": "background-color: #fef2f2;",
            }).fillna("background-color: #fffbeb;")

            def _highlight_row(row):
                return [row_colors.iloc[row.name]] * len(row)

            styled_hasil = hasil_sorted.style.apply(_highlight_row, axis=1)

            st.dataframe(
                styled_hasil,
                use_container_width=True,
                column_config={
                    "Profil": st.column_config.TextColumn("Profil", width="large"),
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                }
            )
            csv = hasil_sorted.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "hasil_fcm.csv", "text/csv", use_container_width=True)

        with tab2:
            summary = hasil.groupby('Profil')[FEATURES].mean().round(4)
            st.dataframe(summary, use_container_width=True)

        with tab3:
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            # Gradien warna hijau -> kuning -> merah mengikuti urutan risk cluster
            cmap_risk = plt.cm.RdYlGn_r
            profiles_by_risk = [cluster_map[c] for c in cluster_risk.sort_values().index.tolist()]
            profiles_by_risk_unique = list(dict.fromkeys(profiles_by_risk))
            n_profiles = len(profiles_by_risk_unique)
            colors_map = {
                label: cmap_risk(i / max(n_profiles - 1, 1)) for i, label in enumerate(profiles_by_risk_unique)
            }
            for profil, color in colors_map.items():
                mask = df_prep['Profil'] == profil
                if mask.any():
                    ax_scatter.scatter(
                        df_prep.loc[mask, 'ROA'], df_prep.loc[mask, 'NPL'],
                        s=150, c=[color], label=profil, alpha=0.85,
                        edgecolors='black', linewidth=1.0
                    )
                    for _, row in df_prep[mask].iterrows():
                        ax_scatter.annotate(row['Ticker'], (row['ROA'], row['NPL']), xytext=(5, 5), textcoords='offset points')
            ax_scatter.set_xlabel("ROA (%)")
            ax_scatter.set_ylabel("NPL (%)")
            ax_scatter.legend()
            ax_scatter.grid(True, alpha=0.3)
            st.pyplot(fig_scatter)

        with tab4:
            heatmap_df = st.session_state['membership'].set_index('Ticker')
            fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_df.iloc[:, :-1], annot=True, cmap="YlGnBu", fmt=".3f", ax=ax_heat)
            st.pyplot(fig_heat)
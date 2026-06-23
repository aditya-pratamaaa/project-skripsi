import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import math

st.set_page_config(
    page_title="FCM Saham Perbankan",
    page_icon="📊",
    layout="wide"
)

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

FEATURES = [
    'ROA',
    'ROE',
    'NPM',
    'DER',
    'NPL',
    'Volatilitas',
    'VaR',
    'EPS'
]

DEFAULT_DATA = {
    'Ticker': ['BBCA', 'BBRI', 'MEGA', 'BBNI', 'BRIS', 'BNGA', 'BDMN', 'ARTO', 'BBTN', 'BJBR'],
    'ROA': [3.48, 3.49, 3.47, 2.22, 2.16, 2.32, 1.46, 0.5, 0.92, 1.27],
    'ROE': [22.28, 16.78, 16.68, 13.28, 16.41, 12.39, 7.34, 1.76, 13.25, 13.82],
    'NPM': [48.68, 22.95, 81.57, 29.64, 19.72, 31.6, 16.47, 7.0, 9.56, 14.74],
    'DER': [479.04, 503.89, 5.37, 645.44, 834.23, 578.81, 3.44, 171.61, 14.26, 1002.07],
    'NPL': [1.88, 3.05, 1.45, 2.50, 2.23, 2.36, 2.22, 0.8, 3.28, 1.75],
    'Volatilitas': [0.229857, 0.284682, 0.338956, 0.297070, 0.445323, 0.229441,  0.241752 , 0.623265 , 0.338190 , 0.184378],
    'VaR': [ 0.021325, 0.026631, 0.027300, 0.027463,  0.039135, 0.019356, 0.021486,   0.054857, 0.030279,  0.017011],
    'EPS': [378.40, 349.60, 14.07, 548.40, 123.16, 253.17, 324.0, 8.41, 244.8, 166.86],
}

n_clusters = 3
m_fuzzy = 2.0
max_iter = 1000
error_tol = 0.005

st.title("📊 Klasterisasi Risiko & Kinerja Saham Perbankan")
st.caption("Metode Fuzzy C-Means (FCM)")

with st.sidebar:
    st.subheader("Parameter")
    st.metric("Jumlah Cluster", 3)
    st.metric("Fuzziness Exponent", m_fuzzy)
    st.metric("Max Iteration", max_iter)
    st.metric("Error Tolerance", error_tol)

tab1, tab2, tab3, tab4 = st.tabs([
    "Input Data",
    "Preprocessing",
    "Modeling",
    "Hasil"
])

with tab1:

    df_input = pd.DataFrame(DEFAULT_DATA)

    df_edit = st.data_editor(
        df_input,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn("Kode Saham"),
            "ROA": st.column_config.NumberColumn(format="%.2f"),
            "ROE": st.column_config.NumberColumn(format="%.2f"),
            "NPM": st.column_config.NumberColumn(format="%.2f"),
            "DER": st.column_config.NumberColumn(format="%.2f"),
            "NPL": st.column_config.NumberColumn(format="%.2f"),
            "Volatilitas": st.column_config.NumberColumn(format="%.4f"),
            "VaR": st.column_config.NumberColumn(format="%.4f"),
            "EPS": st.column_config.NumberColumn(format="%.1f"),
        }
    )

    st.subheader("Statistik Deskriptif")
    st.dataframe(
        df_edit[FEATURES].describe().round(4),
        use_container_width=True
    )

    st.subheader("Korelasi Variabel")

    fig_corr, ax_corr = plt.subplots(figsize=(8, 4))

    sns.heatmap(
        df_edit[FEATURES].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax_corr
    )

    st.pyplot(fig_corr)

    st.session_state['df_main'] = df_edit.copy()

with tab2:

    df_prep = st.session_state['df_main'].copy()

    if len(df_prep) < n_clusters + 2:
        st.error(f"Minimal data {n_clusters + 2}")
        st.stop()

    missing = df_prep[FEATURES].isnull().sum().sum()

    col1, col2 = st.columns(2)

    col1.metric("Jumlah Data", len(df_prep))
    col2.metric("Missing Value", int(missing))

    if missing > 0:
        df_prep[FEATURES] = df_prep[FEATURES].fillna(
            df_prep[FEATURES].median()
        )

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(df_prep[FEATURES])

    df_norm = pd.DataFrame(
        data_scaled,
        columns=FEATURES
    )

    df_norm.insert(0, 'Ticker', df_prep['Ticker'].values)

    st.subheader("Data Normalisasi")

    st.dataframe(
        df_norm.round(4),
        use_container_width=True
    )

    st.session_state['data_scaled'] = data_scaled
    st.session_state['df_prep'] = df_prep

with tab3:

    data_scaled = st.session_state['data_scaled']
    df_prep = st.session_state['df_prep']

    data_T = data_scaled.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_T,
        c=n_clusters,
        m=m_fuzzy,
        error=error_tol,
        maxiter=max_iter
    )

    cluster_labels = np.argmax(u, axis=0)

    col1, col2, col3 = st.columns(3)

    col1.metric("Cluster", n_clusters)
    col2.metric("FPC", round(fpc, 4))
    col3.metric("Iterasi", p)

    membership = pd.DataFrame(
        u.T,
        columns=[f"Cluster {i+1}" for i in range(n_clusters)]
    )

    membership.insert(0, 'Ticker', df_prep['Ticker'].values)
    membership['Cluster'] = cluster_labels + 1

    st.subheader("Membership Degree")

    st.dataframe(
        membership.round(4),
        use_container_width=True
    )

    fig_jm, ax_jm = plt.subplots(figsize=(8, 3))

    ax_jm.plot(jm, linewidth=2)

    ax_jm.set_title("Konvergensi FCM")

    st.pyplot(fig_jm)

    st.session_state['cluster_labels'] = cluster_labels
    st.session_state['membership'] = membership
    st.session_state['fpc'] = fpc

with tab4:

    df_prep = st.session_state['df_prep'].copy()

    cluster_labels = st.session_state['cluster_labels']

    df_prep['Cluster'] = cluster_labels

    cluster_map = {
        0: "🟢 Low Risk — High Return",
        1: "🟡 Medium Risk — Medium Return",
        2: "🔴 High Risk — Low Return"
    }

    df_prep['Profil'] = df_prep['Cluster'].map(cluster_map)

    st.subheader("Hasil Klasterisasi")

    hasil = df_prep.copy()

    hasil['Cluster'] = hasil['Cluster'] + 1

    st.dataframe(
        hasil,
        use_container_width=True
    )

    st.subheader("Rata-rata per Cluster")

    summary = hasil.groupby('Profil')[FEATURES].mean().round(4)

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.subheader("Visualisasi Cluster")

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))

    colors = ['green', 'orange', 'red']

    for i in range(n_clusters):

        mask = cluster_labels == i

        ax_scatter.scatter(
            df_prep.loc[mask, 'ROA'],
            df_prep.loc[mask, 'NPL'],
            s=120,
            label=f"Cluster {i+1}"
        )

        for ticker, x, y in zip(
            df_prep.loc[mask, 'Ticker'],
            df_prep.loc[mask, 'ROA'],
            df_prep.loc[mask, 'NPL']
        ):
            ax_scatter.annotate(
                ticker,
                (x, y)
            )

    ax_scatter.set_xlabel("ROA")
    ax_scatter.set_ylabel("NPL")
    ax_scatter.legend()

    st.pyplot(fig_scatter)

    st.subheader("Heatmap Membership")

    heatmap_df = st.session_state['membership'].set_index('Ticker')

    fig_heat, ax_heat = plt.subplots(figsize=(8, 4))

    sns.heatmap(
        heatmap_df.iloc[:, :-1],
        annot=True,
        cmap="YlGnBu",
        fmt=".3f",
        ax=ax_heat
    )

    st.pyplot(fig_heat)

    csv = hasil.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="hasil_fcm.csv",
        mime="text/csv"
    )

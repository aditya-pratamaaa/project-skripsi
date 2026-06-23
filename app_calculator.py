import streamlit as st
import pandas as pd

# --- Fungsi Backend Kalkulasi Rasio ---
def calculate_ratios(net_income, total_assets, total_equity):
    """Menghitung ROA dan ROE."""
    
    # Menghindari pembagian dengan nol
    if total_assets == 0:
        roa = 0.0
    else:
        # ROA = (Laba Bersih / Total Aset) * 100
        roa = (net_income / total_assets) * 100
        
    if total_equity == 0:
        roe = 0.0
    else:
        # ROE = (Laba Bersih / Total Ekuitas) * 100
        roe = (net_income / total_equity) * 100
        
    return roa, roe

# --- UI Streamlit ---
def main():
    st.set_page_config(layout="wide")
    st.title("🧮 Kalkulator Rasio Kinerja Bank Sederhana")
    st.caption("Masukkan nilai fundamental bank untuk melihat rasio ROA dan ROE.")
    
    # Buat dua kolom untuk input dan output
    col1, col2 = st.columns(2)

    # --- Kolom 1: Input Data Mentah ---
    with col1:
        st.header("1. Input Data Finansial")
        st.info("Masukkan nilai dalam satuan mata uang (misalnya, Rupiah atau Juta Rupiah).")
        
        # Menggunakan st.number_input untuk input numerik
        net_income = st.number_input("Laba Bersih (Net Income)", min_value=0.0, format="%.2f")
        total_assets = st.number_input("Total Aset (Total Assets)", min_value=0.0, format="%.2f")
        total_equity = st.number_input("Total Ekuitas (Total Equity)", min_value=0.0, format="%.2f")
        
    # --- Kolom 2: Hasil Kalkulasi (Real-Time) ---
    with col2:
        st.header("2. Hasil Rasio")
        
        # Panggil fungsi kalkulasi
        roa, roe = calculate_ratios(net_income, total_assets, total_equity)
        
        # Tampilkan Hasil Menggunakan st.metric
        st.metric(
            label="Return on Assets (ROA)", 
            value=f"{roa:.2f} %",
            delta="Mengukur efisiensi penggunaan aset."
        )
        
        st.metric(
            label="Return on Equity (ROE)", 
            value=f"{roe:.2f} %",
            delta="Mengukur profitabilitas modal pemegang saham."
        )

        st.subheader("Interpretasi Cepat:")
        if roa > 1.5 and roe > 10:
            st.success("Kinerja Kategori Baik: Profitabilitas Tinggi.")
        elif roa > 0:
            st.warning("Kinerja Moderat: Profitabilitas Terdeteksi.")
        else:
            st.error("Kinerja Kategori Kurang Baik: Belum Mencapai Laba.")

if __name__ == "__main__":
    main()
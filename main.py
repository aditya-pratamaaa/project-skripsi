import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --- DEFINISI GLOBAL ---
# Variabel yang akan digunakan untuk klasterisasi (total 8 variabel)
CLUSTER_VARS = ['ROA', 'ROE', 'NPM', 'DER', 'NPL', 'VOLATILITY', 'VaR', 'EPS']
NUM_CLUSTERS = 3  # Tentukan jumlah klaster awal (misalnya, Low/Medium/High Risk-Return)
FUZZINESS_M = 2.0 # Parameter fuzziness m (umumnya 2.0)

# --- 1. FUNGSI PERHITUNGAN RISIKO PASAR (VOLATILITY & VaR) ---
def calculate_market_risk(price_file):
    """Menghitung Volatilitas Tahunan dan VaR Tahunan 99%."""
    price_df = pd.read_csv(price_file, index_col='Date', parse_dates=True)
    
    # Hitung Return Logaritmik
    returns = np.log(price_df / price_df.shift(1)).dropna()
    
    # Volatilitas Tahunan (Std Dev * sqrt(252))
    volatility = returns.std() * np.sqrt(252)
    
    # VaR Tahunan (Historical Simulation 99%)
    var_daily = -returns.quantile(0.01) 
    var_annual = var_daily * np.sqrt(252)
    
    risk_df = pd.DataFrame({
        'VOLATILITY': volatility,
        'VaR': var_annual
    }).reset_index().rename(columns={'index': 'Ticker'})
    
    return risk_df

# --- 2. FUNGSI PREPARASI DATA & NORMALISASI ---
def prepare_data(ratio_file, risk_df):
    """Menggabungkan data rasio, risiko, dan melakukan normalisasi."""
    
    # Muat Data Rasio Keuangan (Asumsi sudah ada Ticker, ROA, ROE, dll.)
    ratio_df = pd.read_csv(ratio_file)
    
    # Gabungkan Data Kinerja dan Risiko
    merged_df = pd.merge(ratio_df, risk_df, on='Ticker', how='inner')
    
    # Pilih dan Normalisasi Variabel Klasterisasi
    X_data = merged_df[CLUSTER_VARS].copy()
    
    # Inisialisasi MinMaxScaler dan terapkan normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    return merged_df['Ticker'], X_scaled

# --- 3. FUNGSI MODELING (FUZZY C-MEANS) ---
def run_fcm_clustering(X_scaled, n_clusters, m_fuzziness):
    """Menerapkan algoritma Fuzzy C-Means."""
    
    # Data harus ditransposisi (fitur = baris, sampel = kolom) untuk skfuzzy
    data_T = X_scaled.T
    
    # Jalankan FCM
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_T, c=n_clusters, m=m_fuzziness, error=0.005, maxiter=1000
    )
    
    # 'u' adalah Matriks Keanggotaan (Membership Matrix)
    return u, cntr, fpc

# --- FUNGSI UTAMA (MAIN) ---
def main():
    print("--- START: KLATERISASI SAHAM BANK DENGAN FUZZY C-MEANS ---")
    
    # 1. DATA UNDERSTANDING & PERHITUNGAN RISIKO
    print("\n[STEP 1/4] Menghitung Volatilitas & VaR...")
    risk_df = calculate_market_risk('data/price_history.csv')
    print("   Data Risiko berhasil dihitung.")
    
    # 2. DATA PREPARATION
    print("\n[STEP 2/4] Menggabungkan & Normalisasi Data...")
    tickers, X_scaled = prepare_data('data/financial_ratios.csv', risk_df)
    print(f"   Data Siap Klasterisasi. Jumlah Sampel: {X_scaled.shape[0]}, Fitur: {X_scaled.shape[1]}")
    
    # 3. MODELING (FCM)
    print("\n[STEP 3/4] Menerapkan Fuzzy C-Means...")
    u, cntr, fpc = run_fcm_clustering(X_scaled, NUM_CLUSTERS, FUZZINESS_M)
    print(f"   FCM Selesai. FPC (Fuzzy Partition Coefficient) = {fpc:.4f}")
    
    # 4. EVALUATION & HASIL AKHIR
    # Tentukan klaster akhir berdasarkan derajat keanggotaan tertinggi
    cluster_labels = u.T.argmax(axis=1) 
    
    results = pd.DataFrame({
        'Ticker': tickers,
        'Cluster': cluster_labels
    })
    
    print("\n[STEP 4/4] Hasil Klasterisasi Sederhana:")
    print(results.head(10))
    print(f"\nDistribusi Klaster:\n{results['Cluster'].value_counts()}")

    # Contoh Simpan Hasil
    results.to_csv('output/fcm_simple_result.csv', index=False)
    print("\nHasil disimpan ke output/fcm_simple_result.csv")

if __name__ == "__main__":
    main()
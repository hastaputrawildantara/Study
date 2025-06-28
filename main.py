import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Membaca dataset           
df = pd.read_csv('train (1).csv')           

# Menampilkan fitur fitur dari dataset
print("===Daftar Fitur=== \n", df.columns.tolist(), "\n")

# Menampilkan 5 baris pertama dari dataset
print("===Isi 5 Baris Pertama===\n", df.head(), "\n")

# Menampilkan informasi dari dataset
print("===Informasi Dataset===\n", df.info(), "\n")

# Menampilkan statistik deskriptif dari dataset
print("===Statistik Deskriptif===\n", df.describe(), "\n")

# Menampilkan jumlah data yang hilang pada setiap kolom
print("===Jumlah Data Kosong===\n", df.isnull().sum(), "\n")

# Menghapus Data Duplikat
duplicates = df.duplicated().sum()
print("Jumlah data duplikat: ", duplicates)
if duplicates > 0:
    df = df.drop_duplicates()
    print("Data duplikat telah dihapus.\n")
else:
    print("Tidak ada data duplikat yang ditemukan.\n")

# Memeriksa dan menghapus data yang hilang
array_object = df.select_dtypes(include='object').columns
array_numeric = df.select_dtypes(include='number').columns
for col in df.columns:
    if col in array_numeric:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    elif col in array_object:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)

#Visualisasi Data
plt.figure(figsize=(8,6))
sns.histplot(df['SalePrice'], kde=True, bins=20)
plt.title('Distribusi Harga Penjualan Rumah')
plt.xlabel('Harga Penjualan')
plt.ylabel('Frekuensi')
plt.show()

# mendeteksi dan menghapus outlier menggunakan IQR
def remove_outliers_iqr(data, cols):
    data_clean = data.copy()
    for col in cols:
        Q1 = data_clean[col].quantile(0.25)
        Q3 = data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier = data_clean[(data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)].index
        print(f"Outlier pada kolom {col}: {len(outlier)} \n")
        data_clean = data_clean.drop(index=outlier)
    return data_clean

# Menghapus Outlier pada kolom numerik
df_no_outliers = remove_outliers_iqr(df, array_numeric)
print(df.shape)
print(df_no_outliers.shape, "\n")

# Normalisasi Data (MinMaxScaler)
df_clean_normalized = df_no_outliers.copy()
scaler = MinMaxScaler()
# Identifikasi kolom numerik tanpa outlier
num_col_no_outliers = df_no_outliers.select_dtypes(include='number').columns.tolist()
df_clean_normalized[num_col_no_outliers] = scaler.fit_transform(df_clean_normalized[num_col_no_outliers])

from sklearn.preprocessing import OrdinalEncoder
# Mengkodekan kolom kategorikal
categorical_cols = df_clean_normalized.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder()
print("sebelum encode: ",df_clean_normalized[categorical_cols].head(), "\n")
df_clean_normalized[categorical_cols] = encoder.fit_transform(df_clean_normalized[categorical_cols])
print("setelah encode: ",df_clean_normalized[categorical_cols].head(), "\n")

df_clean_normalized.to_csv('train_clean_normalized.csv', index=False)


# Visualisasi Data Normalisasi
plt.figure(figsize=(8,6))                                               
sns.histplot(df_clean_normalized['SalePrice'], kde=True, bins=20)
plt.title('Distribusi Harga Penjualan Rumah Setelah Normalisasi')
plt.xlabel('Harga Penjualan (Normalisasi)') 
plt.ylabel('Frekuensi')
plt.show()


# Train dan Test Split
from sklearn.model_selection import train_test_split
X = df_clean_normalized.drop(columns=['SalePrice'])
y = df_clean_normalized['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)
print("Jumlah data training: ", len(X_train), "\n")
print("Jumlah data Testing: ", len(X_test), "\n")


# Memilih Fitur
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
print("Fitur yang dipilih: ", X_train.columns[selector.get_support()].tolist())


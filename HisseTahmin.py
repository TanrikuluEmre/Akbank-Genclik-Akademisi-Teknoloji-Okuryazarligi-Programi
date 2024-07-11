# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tWmLj-sVxFeZW80sGGm15LM3e4vzRfcv
"""

pip install yfinance

import yfinance as yf
import pandas as pd

# Hisse senedi sembolü
ticker_symbol = 'AAPL'  # Örneğin Apple Inc.

# Hisse senedi verilerini al
ticker_data = yf.Ticker(ticker_symbol)

# Geçmiş 5 yıllık günlük verileri al
hist_data = ticker_data.history(period='5y')

# Verileri CSV dosyasına kaydet
csv_file_path = 'hisse_senedi_verileri.csv'
hist_data.to_csv(csv_file_path)

print(f"Hisse senedi verileri {csv_file_path} dosyasına kaydedildi.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Veri setini yükle
csv_file_path = 'hisse_senedi_verileri.csv'
data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)

# Sadece 'Open' ve 'Close' kolonlarını kullanacağız
data = data[['Open', 'Close']]

# Özellik ve hedef değişkeni oluştur
data['Open_lag1'] = data['Open'].shift(-1)  # Bir gün sonraki açılış fiyatı
data.dropna(inplace=True)

# Veriyi ölçeklendirme
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Veriyi yeniden şekillendirme
X = data_scaled[:-1, :-1]  # Son satırı hariç tüm satırlar ve tüm sütunlar (hedef değişken hariç)
y = data_scaled[:-1, -1]  # Son sütun (hedef değişken)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # LSTM için veriyi 3 boyutlu hale getirme

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM modelini oluşturma
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

# Modeli derleme
model.compile(optimizer='adam', loss='mse')

# Modeli eğitme
model.fit(X_train, y_train, epochs=50, verbose=1)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Tahminleri geri ölçeklendirme
y_test_rescaled = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], -1), y_test.reshape(-1, 1))))[:, -1]
y_pred_rescaled = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], -1), y_pred)))[:, -1]

# Model performansını değerlendirme
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Sonuçları karşılaştırma
results = pd.DataFrame({'Actual': y_test_rescaled, 'Predicted': y_pred_rescaled})
print(results.head())
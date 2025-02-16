# 📌 Import pustaka yang dibutuhkan
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files  # Digunakan untuk upload dataset di Google Colab

# 1️⃣ **UPLOAD DATASET (KHUSUS UNTUK GOOGLE COLAB)**
print("Silakan upload file dataset (bankloans.csv)")
uploaded = files.upload()  # Tunggu hingga file diunggah

# 2️⃣ **BACA DATASET YANG DIUNGGAH**
file_name = list(uploaded.keys())[0]  # Ambil nama file yang diupload
df = pd.read_csv(file_name)  # Baca file CSV

# 3️⃣ **CEK MISSING VALUES & HAPUS DATA KOSONG**
df = df.dropna()  # Hapus baris dengan NaN

# 4️⃣ **PILIH FITUR & TARGET**
features = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt']
target = 'default'

X = df[features].values
y = df[target].values

# 5️⃣ **NORMALISASI DATA**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6️⃣ **GUNAKAN SMOTE UNTUK IMBALANCED DATA**
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 7️⃣ **BAGI DATA MENJADI TRAINING & TESTING (80:20)**
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 8️⃣ **MEMBANGUN MODEL MLFN**
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden layer pertama
    Dropout(0.2),  # Dropout untuk mencegah overfitting
    Dense(16, activation='relu'),  # Hidden layer kedua
    Dropout(0.1),
    Dense(8, activation='tanh'),  # Hidden layer ketiga
    Dense(1, activation='sigmoid')  # Output layer (Sigmoid untuk klasifikasi biner)
])

# 9️⃣ **KOMPILASI MODEL**
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔟 **TRAINING MODEL**
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 🔟 **EVALUASI MODEL**
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# 🔟 **HASIL EVALUASI**
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔟 **VISUALISASI HASIL TRAINING**
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# 🔟 **VISUALISASI HUBUNGAN ANTAR VARIABEL (Income vs Debt Ratio)**
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['income'], y=df['debtinc'], hue=df['default'])
plt.title("Hubungan Income & Debt Ratio terhadap Risiko Default")
plt.xlabel("Income")
plt.ylabel("Debt-to-Income Ratio")
plt.show()alance
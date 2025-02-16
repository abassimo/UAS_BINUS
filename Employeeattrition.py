# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# Upload dataset dari Google Colab
from google.colab import files
uploaded = files.upload()

# Membaca dataset yang diunggah
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# Pilih variabel yang digunakan
selected_features = ['OverTime', 'YearsAtCompany', 'TotalWorkingYears', 'MonthlyIncome']
target_variable = 'Attrition'

# Konversi target variabel 'Attrition' ke numerik (Yes=1, No=0)
df[target_variable] = df[target_variable].apply(lambda x: 1 if x == "Yes" else 0)

# Konversi variabel kategorikal 'OverTime' ke numerik (Yes=1, No=0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)

# Normalisasi variabel numerik
scaler = StandardScaler()
df[selected_features[1:]] = scaler.fit_transform(df[selected_features[1:]])

# Pisahkan dataset menjadi training dan testing
X = df[selected_features]
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inisialisasi model
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=30, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Dictionary untuk menyimpan hasil evaluasi
results = {}

# Training dan evaluasi model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Simpan hasil evaluasi
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

# Konversi hasil evaluasi ke dalam DataFrame
results_df = pd.DataFrame(results).T

# Visualisasi hasil evaluasi model
plt.figure(figsize=(10, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
for metric in metrics:
    plt.plot(results_df.index, results_df[metric], marker='o', label=metric)
plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Perbandingan Performa Model")
plt.legend()
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Menampilkan hasil evaluasi
display(results_df)

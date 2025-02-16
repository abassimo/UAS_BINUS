# 📌 Import pustaka yang dibutuhkan
import pandas as pd
import re
import string
import nltk
import shutil  # Untuk menghapus cache NLTK jika bermasalah
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from google.colab import files  # Untuk upload & download dataset di Google Colab
from tqdm import tqdm  # Untuk progress bar

# 🔹 Coba hapus cache NLTK jika sebelumnya bermasalah
try:
    shutil.rmtree("/root/nltk_data")  # Hapus cache NLTK
    print("Cache NLTK dihapus, mengunduh ulang resource...")
except FileNotFoundError:
    print("Tidak ada cache NLTK yang perlu dihapus.")

# 🔹 Download ulang resource NLTK
nltk.download('punkt')
nltk.download('stopwords')
# Ensure the 'stopwords' dataset is downloaded
nltk.download('stopwords') # Download the stopwords dataset

nltk.download('punkt_tab')  # Tambahan jika tetap error

# Cek apakah resource sudah terunduh dengan benar
try:
    print("Cek resource punkt:", nltk.data.find('tokenizers/punkt'))
except LookupError:
    print("Resource punkt tidak ditemukan, harap ulangi pengunduhan!")

# 1️⃣ **UPLOAD DATASET (KHUSUS UNTUK GOOGLE COLAB)**
print("Silakan upload file dataset (contoh: reviews.csv)")
uploaded = files.upload()  # Tunggu hingga file diunggah

# 2️⃣ **BACA DATASET YANG DIUNGGAH**
file_name = list(uploaded.keys())[0]  # Ambil nama file yang diupload
df = pd.read_csv(file_name, encoding="latin1")  # Membaca file CSV

# 3️⃣ **LIHAT INFORMASI DATASET**
print("\n🔹 Informasi Dataset:")
print(df.info())

# 4️⃣ **PILIH KOLOM TEKS UNTUK DIPROSES**
if 'Review' not in df.columns:
    raise KeyError("Kolom 'Review' tidak ditemukan dalam dataset. Pastikan nama kolom benar.")

df = df[['Review']].dropna()  # Menghapus baris yang memiliki nilai kosong (NaN)

# 5️⃣ **Inisialisasi stopwords dan stemmer hanya sekali (lebih efisien)**
# Download the 'stopwords' resource if it hasn't been downloaded already.
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# 6️⃣ **FUNGSI PEMROSESAN TEKS (Optimal & Lebih Cepat)**
def preprocess_text(text):
    # **1️⃣ Case Folding**: Ubah semua teks menjadi huruf kecil
    text = text.lower()
    
    # **2️⃣ Removing Punctuation**: Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # **3️⃣ Tokenization & Removing Stop Words**: Tokenisasi dan hapus stopwords
    words = word_tokenize(text)  # 🚀 Pakai tokenizer yang sudah diunduh
    filtered_words = [word for word in words if word not in stop_words]
    
    # **4️⃣ Stemming**: Ubah kata ke bentuk dasarnya
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    return " ".join(stemmed_words)

# 7️⃣ **APLIKASIKAN PEMROSESAN TEKS KE ULASAN (Dengan Progress Bar)**
tqdm.pandas()
df['cleaned_review'] = df['Review'].astype(str).progress_apply(preprocess_text)

# 8️⃣ **LIHAT HASIL SETELAH PREPROCESSING**
print("\n🔹 Contoh Hasil Preprocessing:")
print(df[['Review', 'cleaned_review']].head())

# 9️⃣ **SIMPAN DATASET YANG TELAH DIPROSES**
df.to_csv("cleaned_reviews.csv", index=False)

print("\n✅ Pemrosesan teks selesai! Dataset yang telah diproses disimpan sebagai 'cleaned_reviews.csv'")

# 🔟 **TOMBOL DOWNLOAD FILE**
files.download("cleaned_reviews.csv")  # Otomatis menampilkan tombol download di Google Colab
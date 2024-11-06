import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Fungsi untuk tahapan preprocessing
def show_preprocessing():
    st.title("Tahapan Preprocessing")

    # Memuat data asli
    original_df = pd.read_csv("2_Kategoriberita-CNN.csv")
    st.write("## Data Asli:")
    st.write(original_df[['judul', 'isi', 'tanggal', 'kategori']].head())  # Menampilkan kolom yang relevan dari dataset asli

    # Memuat data yang sudah dibersihkan
    df = pd.read_csv("preprocessing-cnnnews.csv")
    st.write("## Data yang telah dibersihkan:")
    st.write(df[['berita_clean', 'case_folding', 'tokenize', 'stopword_removal']].head())

    # Pastikan kolom 'stopword_removal' berisi string
    df['stopword_removal'] = df['stopword_removal'].astype(str)

    # Memisahkan data menjadi data latih dan data uji
    X = df['stopword_removal']
    y = df['kategori']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menampilkan hasil TF-IDF untuk data training dan testing
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit-transform pada data latih
    X_test_tfidf = vectorizer.transform(X_test)  # Transform pada data uji

    # Menampilkan hasil TF-IDF Data Training
    st.write("## Hasil TF-IDF Data Training:")
    X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    st.write(X_train_tfidf_df.head())  # Menampilkan beberapa baris pertama

    # Menampilkan hasil TF-IDF Data Testing
    st.write("## Hasil TF-IDF Data Testing:")
    X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    st.write(X_test_tfidf_df.head())  # Menampilkan beberapa baris pertama

    # Proses Reduksi Dimensi dengan SVD (Singular Value Decomposition)
    svd = TruncatedSVD(n_components=100)  # Anda bisa menyesuaikan jumlah komponen
    X_train_svd = svd.fit_transform(X_train_tfidf)  # Mengurangi dimensi data latih
    X_test_svd = svd.transform(X_test_tfidf)  # Mengurangi dimensi data uji

    # Menampilkan hasil setelah SVD
    st.write("## Hasil Reduksi Dimensi (SVD) Data Training:")
    X_train_svd_df = pd.DataFrame(X_train_svd)
    st.write(X_train_svd_df.head())  # Menampilkan beberapa baris pertama setelah reduksi dimensi

    st.write("## Hasil Reduksi Dimensi (SVD) Data Testing:")
    X_test_svd_df = pd.DataFrame(X_test_svd)
    st.write(X_test_svd_df.head())  # Menampilkan beberapa baris pertama setelah reduksi dimensi

    # Melatih model Logistic Regression dengan data yang telah direduksi dimensi
    model = LogisticRegression()
    model.fit(X_train_svd, y_train)

    # Memprediksi data uji
    y_pred = model.predict(X_test_svd)

    # Evaluasi model dan menampilkan akurasi, precision, recall, dan f1-score dalam persen
    accuracy = accuracy_score(y_test, y_pred) * 100  # Menghitung akurasi dalam persen
    st.write(f"**Akurasi Model**: {accuracy}%")

    # Menampilkan precision, recall, dan f1-score dalam format persen
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['accuracy'] * 100 # Precision
    recall = report['macro avg']['recall'] * 100  # Recall
    f1 = report['macro avg']['f1-score'] * 100 # F1-score

    st.write(f"**Precision**: {precision}%")
    st.write(f"**Recall**: {recall}%")
    st.write(f"**F1-Score**: {f1}%")

# Menampilkan halaman
if __name__ == "__main__":
    show_preprocessing()

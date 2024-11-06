import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Memuat data
df = pd.read_csv("preprocessing-cnnnews.csv")
df['stopword_removal'] = df['stopword_removal'].astype(str)

X = df['stopword_removal']
y = df['kategori']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menghitung TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit-transform pada data latih
X_test_tfidf = vectorizer.transform(X_test)  # Transform pada data uji

# Reduksi Dimensi dengan SVD
svd = TruncatedSVD(n_components=100)  # Menentukan jumlah komponen untuk reduksi dimensi
X_train_svd = svd.fit_transform(X_train_tfidf)  # Mengurangi dimensi data latih
X_test_svd = svd.transform(X_test_tfidf)  # Mengurangi dimensi data uji

# Melatih model Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_svd, y_train)

# Memprediksi data uji
y_pred = logreg.predict(X_test_svd)

# Evaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Menyimpan model dan vectorizer
joblib.dump(logreg, "logreg_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(svd, "svd_model.pkl")  # Menyimpan model SVD

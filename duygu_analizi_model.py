import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re

data = pd.read_csv('IMDB Dataset.csv')
#print("Veri Setinin İlk 5 Satırı:")
#print(data.head())
#print("\nVeri Seti Bilgileri:")
#print(data.info())

# 'positive' -> 1, 'negative' -> 0 olarak dönüştürme
data['sentiment'] = data['sentiment'].apply(
    lambda x: 1 if x == 'positive' else 0
)

#print("\nDönüştürülmüş Hedef Değişkenin İlk 5 Satırı:")
#print(data.head())

def clean_text(text):
    # 1. HTML etiketlerini kaldır
    text = re.sub(r'<br\s*/?>', ' ', text)
    
    # 2. Küçük harfe dönüştür
    text = text.lower()
    
    # 3. Alfanümerik olmayan karakterleri ve rakamları kaldır (sadece harf bırak)
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Gereksiz kelimeleri kaldır
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # 5. Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Veri setindeki tüm yorumları temizle
data['review'] = data['review'].apply(clean_text)

# print("\nTemizlenmiş Yorumun Bir Örneği (ilk satır):")
# print(data['review'].iloc[0])

# Veriyi Yorumlar (X) ve Duygular (y) olarak ayırma
X = data['review']
y = data['sentiment']

# Eğitim (%80) ve Test (%20) olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Eğitim seti boyutu:", len(X_train))
print("Test seti boyutu:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vektörleştiriciyi Tanımlama
# max_features: En çok geçen 5000 kelimeyi kullan.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Yalnızca EĞİTİM verisine FIT edip (öğrenip) transform etme
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# TEST verisine sadece transform etme (yeni kelime öğrenmez, sadece eğitimde öğrendiğini uygular)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("Eğitim TF-IDF Matrisi Şekli:", X_train_tfidf.shape)
print("Test TF-IDF Matrisi Şekli:", X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression

# Modeli Tanımlama
model = LogisticRegression(solver='liblinear', random_state=42)

# Modeli Eğitme
print("\nModel Eğitiliyor...")
model.fit(X_train_tfidf, y_train)
print("Model Eğitimi Tamamlandı.")

from sklearn.metrics import accuracy_score, classification_report

# Test veri setinde tahmin yapma
y_pred = model.predict(X_test_tfidf)

# Doğruluk (Accuracy) Puanını Hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Doğruluğu (Test Seti): {accuracy*100:.2f}%")

# Detaylı Performans Raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))
































=======
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re

data = pd.read_csv('IMDB Dataset.csv')
#print("Veri Setinin İlk 5 Satırı:")
#print(data.head())
#print("\nVeri Seti Bilgileri:")
#print(data.info())

# 'positive' -> 1, 'negative' -> 0 olarak dönüştürme
data['sentiment'] = data['sentiment'].apply(
    lambda x: 1 if x == 'positive' else 0
)

#print("\nDönüştürülmüş Hedef Değişkenin İlk 5 Satırı:")
#print(data.head())

def clean_text(text):
    # 1. HTML etiketlerini kaldır
    text = re.sub(r'<br\s*/?>', ' ', text)
    
    # 2. Küçük harfe dönüştür
    text = text.lower()
    
    # 3. Alfanümerik olmayan karakterleri ve rakamları kaldır (sadece harf bırak)
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Gereksiz kelimeleri kaldır
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # 5. Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Veri setindeki tüm yorumları temizle
data['review'] = data['review'].apply(clean_text)

print("\nTemizlenmiş Yorumun Bir Örneği (ilk satır):")
print(data['review'].iloc[0])
>>>>>>> 2cb965d5aaeb2de1ef8937a5c1a03f416d9d6857

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
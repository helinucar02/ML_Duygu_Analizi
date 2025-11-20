{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "750c7ed0-57ac-481f-90d8-073b3ef87fab",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from nltk.corpus import stopwords\n",
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e0ff661c-e994-471b-99bc-7a144795a19c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Veri Setinin İlk 5 Satırı:\n",
      "                                              review sentiment\n",
      "0  One of the other reviewers has mentioned that ...  positive\n",
      "1  A wonderful little production. <br /><br />The...  positive\n",
      "2  I thought this was a wonderful way to spend ti...  positive\n",
      "3  Basically there's a family where a little boy ...  negative\n",
      "4  Petter Mattei's \"Love in the Time of Money\" is...  positive\n",
      "\n",
      "Veri Seti Bilgileri:\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 100 entries, 0 to 99\n",
      "Data columns (total 2 columns):\n",
      " #   Column     Non-Null Count  Dtype \n",
      "---  ------     --------------  ----- \n",
      " 0   review     100 non-null    object\n",
      " 1   sentiment  100 non-null    object\n",
      "dtypes: object(2)\n",
      "memory usage: 1.7+ KB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv('IMDB Dataset.csv')\n",
    "print(\"Veri Setinin İlk 5 Satırı:\")\n",
    "print(data.head())\n",
    "print(\"\\nVeri Seti Bilgileri:\")\n",
    "print(data.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9072a8a8-b45d-47fe-ab8b-050d33b2b6bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Dönüştürülmüş Hedef Değişkenin İlk 5 Satırı:\n",
      "                                              review  sentiment\n",
      "0  One of the other reviewers has mentioned that ...          1\n",
      "1  A wonderful little production. <br /><br />The...          1\n",
      "2  I thought this was a wonderful way to spend ti...          1\n",
      "3  Basically there's a family where a little boy ...          0\n",
      "4  Petter Mattei's \"Love in the Time of Money\" is...          1\n"
     ]
    }
   ],
   "source": [
    "# 'positive' -> 1, 'negative' -> 0 olarak dönüştürme\n",
    "data['sentiment'] = data['sentiment'].apply(\n",
    "    lambda x: 1 if x == 'positive' else 0\n",
    ")\n",
    "\n",
    "print(\"\\nDönüştürülmüş Hedef Değişkenin İlk 5 Satırı:\")\n",
    "print(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4e83a61e-94fe-4ceb-aff1-f55dc05ea478",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Temizlenmiş Yorumun Bir Örneği (ilk satır):\n",
      "one reviewers mentioned watching oz episode youll hooked right exactly happened first thing struck oz brutality unflinching scenes violence set right word go trust show faint hearted timid show pulls punches regards drugs sex violence hardcore classic use word called oz nickname given oswald maximum security state penitentary focuses mainly emerald city experimental section prison cells glass fronts face inwards privacy high agenda em city home manyaryans muslims gangstas latinos christians italians irish moreso scuffles death stares dodgy dealings shady agreements never far away would say main appeal show due fact goes shows wouldnt dare forget pretty pictures painted mainstream audiences forget charm forget romanceoz doesnt mess around first episode ever saw struck nasty surreal couldnt say ready watched developed taste oz got accustomed high levels graphic violence violence injustice crooked guards wholl sold nickel inmates wholl kill order get away well mannered middle class inmates turned prison bitches due lack street skills prison experience watching oz may become comfortable uncomfortable viewingthats get touch darker side\n"
     ]
    }
   ],
   "source": [
    "def clean_text(text):\n",
    "    # 1. HTML etiketlerini kaldır\n",
    "    text = re.sub(r'<br\\s*/?>', ' ', text)\n",
    "    \n",
    "    # 2. Küçük harfe dönüştür\n",
    "    text = text.lower()\n",
    "    \n",
    "    # 3. Alfanümerik olmayan karakterleri ve rakamları kaldır (sadece harf bırak)\n",
    "    text = re.sub(r'[^a-z\\s]', '', text)\n",
    "\n",
    "    # 4. Gereksiz kelimeleri kaldır\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    text = ' '.join(word for word in text.split() if word not in stop_words)\n",
    "    \n",
    "    # 5. Birden fazla boşluğu tek boşluğa indir\n",
    "    text = re.sub(r'\\s+', ' ', text).strip()\n",
    "    \n",
    "    return text\n",
    "\n",
    "# Veri setindeki tüm yorumları temizle\n",
    "data['review'] = data['review'].apply(clean_text)\n",
    "\n",
    "print(\"\\nTemizlenmiş Yorumun Bir Örneği (ilk satır):\")\n",
    "print(data['review'].iloc[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eee3c144-0681-4ecb-b6c9-39bff7b7fe9b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}


# Librerias

import nltk
import pandas as pd
from textwrap import wrap
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Obtener datos

df = pd.read_csv("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/CSV/df_total.csv", encoding = "utf-8")

print("Dataset: \n", df)
print("\n".join(wrap(df["news"][3])))

# Stemming
# Conjuntos de palabras a utilizar

nltk.download("punkt")
nltk.download("stopwords")

stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(texto):

  tokens = word_tokenize(texto.lower())
  stems = [stemmer.stem(token) for token in tokens if token.isalpha()]

  return " ".join(stems)

df["news_stemmer"] = df["news"].apply(tokenize_and_stem)
print(df["news_stemmer"])

# Division de datos

x = df["news_stemmer"]
y = df["Type"]

print("X: \n", x)
print("Y: \n", y)
print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print("X Train: \n", x_train)
print("Cantidad de X Train:", x_train.shape)
print("X Test: \n", x_test)
print("Cantidad de X Test:", x_test.shape)
print("Y Train: \n", y_train)
print("Cantidad de Y Train:", y_train.shape)
print("Y Test: \n", y_test)
print("Cantidad de Y Test:", y_test.shape)

# Vectorizacion de datos

vectorizer = CountVectorizer()

x_train_vectorizado = vectorizer.fit_transform(x_train)
x_test_vectorizado = vectorizer.transform(x_test)

# Modelo
modelo = MultinomialNB()

# Entrenamiento
modelo.fit(x_train_vectorizado, y_train)

# Prediccion
y_pred = modelo.predict(x_test_vectorizado)

print("Prediccion:", y_pred)
print("Metrica:", metrics.accuracy_score(y_test, y_pred))

# !python -m spacy download es_core_news_sm -q

# Librerias

import spacy
import pandas as pd
from textwrap import wrap
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Obtener datos

df = pd.read_csv("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/CSV/df_total.csv", encoding = "utf-8")

print("Dataset: \n", df)
print("\n".join(wrap(df["news"][3])))

# Lemmatization
# Modelo

nlp = spacy.load("es_core_news_sm")

def Lemmatize_text(texto):

  doc = nlp(texto.lower())
  lemmas = [token.lemma_ for token in doc if token.is_alpha]
  return " ".join(lemmas)

df["news_lemma"] = df["news"].apply(Lemmatize_text)
print(df["news_lemma"])

# Division de datos

x = df["news_lemma"]
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

x_train_transformados = vectorizer.fit_transform(x_train)
x_test_transformado = vectorizer.transform(x_test)

# Modelo
modelo = MultinomialNB()

# Entrenamiento
modelo.fit(x_train_transformados, y_train)

# Prediccion
y_pred = modelo.predict(x_test_transformado)

# Metrica
print(metrics.accuracy_score(y_test, y_pred))
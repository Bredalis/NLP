
# Librerias

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Obtener datos

df = pd.read_csv("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/CSV/spam.csv", encoding = "ISO-8859-1")
cantidad_etiquetas = df.groupby("label").count()

print(f"DF: \n{df.head()}")
print(f"Cantidad de etiquetas: \n{cantidad_etiquetas}")

# Nueva columna con valores binarios

df["b_labels"] = df["label"].map({"ham": 0, "spam": 1})
y = df["b_labels"].to_numpy()

# Dividir los datos

x_train, x_test, y_train, y_test = train_test_split(df["email"], y, test_size = 0.33)

print(f"x train: \n{x_train}")
print(f"x test: \n{x_test}")
print(f"y train: \n{y_train}")
print(f"y test: \n{y_test}")

# Combertir a vectores 

vectores = CountVectorizer(decode_error = "ignore")

x_train = vectores.fit_transform(x_train)
x_test = vectores.transform(x_test)

# Modelo

model = MultinomialNB()
model.fit(x_train, y_train)

print(f"TRAIN SCORE: {model.score(x_train, y_train)}")
print(f"TEST SCORE: {model.score(x_test, y_test)}")

# Crear columna de predicciones

x = vectores.transform(df["email"])
df["predicciones"] = model.predict(x)

print(df)

# Grafica

df["label"].hist()
plt.title("NOSPAM / SPAM")

plt.show()
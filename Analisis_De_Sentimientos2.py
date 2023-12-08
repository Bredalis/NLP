
# Librerias

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# Obtener datos

df = pd.read_excel("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/BBDD.xlsx")
print(f"DF: {df.head()}")
# print(df["sentimiento"] == "negativo")

# DF con sentimiento y review_es como columnas

df = df[["sentimiento", "review_es"]]
print(F"DF nuevo: {df}") 

# Convertir datos a binario

etiqueta_map = {"positivo": 1, "negativo": 0}
df["target"] = df["sentimiento"].map(etiqueta_map)

print(F"DF: {df}") 

# Division de datos

df_train, df_test = train_test_split(df)

print(f"Train DF:{df_train}")
print(f"Test DF:{df_test}")

# Vectorizacion

vectorizer = TfidfVectorizer(max_features = 2000)

x_train = vectorizer.fit_transform(df_train["review_es"])
x_test = vectorizer.transform(df_test["review_es"])

y_train = df_train["target"]
y_test = df_test["target"]

print(x_train)
print(x_test)

print(y_train)
print(y_test)

# Modelo

modelo = LogisticRegression(max_iter = 1000)
modelo.fit(x_train, y_train)

# Metricas evaluativas

print(f"Train acc: {modelo.score(x_train, y_train)}")
print(f"Test acc: {modelo.score(x_test, y_test)}")

# Predicciones

p_train = modelo.predict(x_train)
p_test = modelo.predict(x_test)

print(f"Prediccion de train: {p_train}")
print(f"Prediccion de test: {p_test}")

prueba = ["estuvo muy entretenida la pelicuala", "estuvo terrible la pelicua, me aburrio mucho"]

# Transformar la entrada con el vectorizador

x = vectorizer.transform(prueba)

# Predecir con el modelo

p = modelo.predict(x)

# Obtener la clase del modelo

clases = modelo.classes_

# Mostrar la clase predicha

for i in range(len(prueba)):
	if clases[p_train[i]] == 0:
		print(f"El comentario '{prueba[i]}' es: Negativo")

	else:
		print(f"El comentario '{prueba[i]}' es: Positivo")

# Grafica

df["sentimiento"].hist()
plt.show()
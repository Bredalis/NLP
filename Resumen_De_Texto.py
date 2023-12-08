
# Librerias

import nltk
import textwrap
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Datos

df = pd.read_csv("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/CSV/df_total.csv")
print(f"DF: {df.head()}")

# Guardar noticias aleatoreamente

doc = df["news"].sample()
print(doc.iloc[0])

doc2 = textwrap.fill(doc.iloc[0], replace_whitespace = False, fix_sentence_endings = True)
print(doc2)

# Separar texto

lineas = doc2.split(".")
print(lineas)

# Vectorizacion y tokenizacon

tokenizar = TfidfVectorizer(
	stop_words = stopwords.words("spanish"),
	norm = "l1"
)

x = tokenizar.fit_transform(lineas)
print(x)

# Guardar tamaño de x

filas, columnas = x.shape

for i in range(filas):
	for j in range(columnas):
		print(x[i, j], end = " ") # Imprime el elemento y un espacio en blanco
	print()

def Obtener_Score(tfidf_row):
	x = tfidf_row[tfidf_row != 0]
	return x.mean()

scores = np.zeros(len(lineas))

for i in range(len(lineas)):
	score = Obtener_Score(x[i, :])
	scores[i] = score

print(scores)

sort_idx = np.argsort(-scores)
print(sort_idx)

print("Resumen:")

oraciones = []

for i in range(0, 8):
	oraciones.append([sort_idx[i], scores[sort_idx[i]], lineas[sort_idx[i]]])
	print(f"{scores[sort_idx]}: {lineas[sort_idx[i]]}")

# Ordenar la lista por el primer elemento de cada sublista

oraciones_ord = sorted(oraciones, key = lambda x: x[0])

# Mostrar noticia

for item in oraciones_ord:
	print(item[2])
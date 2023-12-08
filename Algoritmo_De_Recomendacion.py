
# Aplicacion del Metodo TF - IDF
# Recomendardor de peliculas

# Librerias

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Obtener datos

df = pd.read_csv("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/CSV/movie_metadata.csv")

print("Dataset: \n", df)

# Reemplazando caracteres

df["genres"] = df["genres"].str.replace("|", " ")
df["plot_keywords"] = df["plot_keywords"].str.replace("|", " ")

print(df["genres"])
print(df["plot_keywords"])

df["texto"] = df[["genres", "plot_keywords"]].apply(lambda row: ' '.join(row.values.astype(str)), axis = 1)

print("Dataset Actualizado: \n", df)

row = df[["genres", "plot_keywords", "texto"]].iloc[0]
print(row)

# Metodo de recomendacion

tfidf = TfidfVectorizer(max_features = 2000)

x = tfidf.fit_transform(df["texto"])

print("Datos: \n", x)

peliculas = pd.Series(df.index, index = df["movie_title"])
peliculas.index = peliculas.index.str.strip()

print("Peliculas: \n", peliculas)
print(peliculas["Pirates of the Caribbean: At World's End"])

consulta = x[1]
consulta.toarray()
print("Consulta: \n", consulta)

similitud = cosine_similarity(consulta, x)
print("Similitud entre peliculas: \n", similitud)

similitud = similitud.flatten()
print(similitud[1])

# Grafica

plt.plot(similitud)
plt.show()

print((-similitud).argsort())

recomendacion = (-similitud).argsort()[1:11]

print("Recomendaciones: \n", df["movie_title"].iloc[recomendacion])
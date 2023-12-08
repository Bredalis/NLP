
"""
Word2Vec: es un modelo que se
utiliza para aprender
representaciones vectoriales
de palabras. Estas representaciones
pueden capturar muchas propiedades
linguisticas de las palabras,
como su significado semantico,
gramatical y hasta contextual.
"""

# !pip install gensim -q
# !pip install pypdf2 -q

# Librerias

import string
import PyPDF2
from textwrap import wrap
from gensim.models import Word2Vec, KeyedVectors

# Datos

with open("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/Minecraft.txt", "r", encoding = "utf-8") as file:
  documento = file.read()

print("Documento Minecraft: \n")
print("\n".join(wrap(documento)))
print("Longitud del documento:", len(documento))

# Preprocesamiento de datos

"""
El objetivo del procesamiento
es convertir el documento en una
lista de frases y cada frase en una
lista de palabras, eliminando signos
de puntuacion y convirtiendo todo a minusculas
"""

# Dividir el documento en frases, usando el punto como separador

oraciones = documento.split(",")

print("Oraciones: \n", oraciones)
print("Longitud de oraciones:", len(oraciones))
print("Primera oracion: \n", oraciones[5])

oraciones_limpias = []

for oracion in oraciones:

  # Eliminar puntuaion y dividir por espacios
  tokens = oracion.translate(str.maketrans('', '', string.punctuation)).split()

  # Convertir a minusculas
  tokens = [word.lower() for word in tokens if word.isalpha]
  if tokens: # Agregar si hay tokens
    oraciones_limpias.append(tokens)

print(oraciones_limpias[0])

# Entrenamiento del modelo (word2Vec)

modelo = Word2Vec(sentences = oraciones_limpias, vector_size = 500, window = 5, min_count = 1, workers = 8)

def Palabras_Cercanas(palabra):

  palabras_cercanas = modelo.wv.most_similar(palabra, topn = 10)
  print(palabras_cercanas)

Palabras_Cercanas("minecraft")
Palabras_Cercanas("hombre")
Palabras_Cercanas("java")

# Guardar y cargar modelo

modelo.save("minecraft.model")
modelo_cargado = Word2Vec.load("minecraft.model")

# Uso del modelo cargado

def Palabras_Cercanas(palabra):

  palabras_cercanas = modelo_cargado.wv.most_similar(palabra, topn = 10)
  print(palabras_cercanas)

Palabras_Cercanas("minecraft")
Palabras_Cercanas("java")

# Guardar y cargar Embeddings

modelo.wv.save_word2vec_format('mine_emb.txt', binary = False)
modelo.wv.save_word2vec_format('mine_emb.bin', binary = True)

embeddings_cargados = KeyedVectors.load_word2vec_format('mine_emb.txt', binary = False)
# O si fue guardado en formato binario:
# embeddings_cargados = KeyedVectors.load_word2vec_format('embeddings.bin', binary = True)
print(embeddings_cargados)

def Analogia(v1, v2, v3):
  similitud = embeddings_cargados.most_similar(positive = [v1, v3], negative = [v2])
  print(f"{v1} es a {v2} como {similitud[0][0]} es a {v3}")

Analogia("jugador", "hombre", "mujer")
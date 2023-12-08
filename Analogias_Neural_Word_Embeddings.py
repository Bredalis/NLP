
"""
Neural Word Embeddings:
son representaciones 
numéricas de palabras. 
Estas incrustaciones 
capturan las relaciones 
semánticas y contextuales 
entre las palabras en un 
espacio multidimensional, 
lo que permite que las palabras 
con significados similares estén 
más cerca entre sí en este espacio. 
"""

# !pip install gensim -q

# Libreria

import gensim

# Datos

vectores = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/SBW-vectors-300-min5.txt")

def analogias(v1, v2, v3):
	similitud = vectores.most_similar(positive = [v1, v3], negative = [v2])
	print(f"{v1} es a {v2} como {similitud[0][0]} es a {v3}")

analogias("rey", "hombre", "mujer")
analogias("musica", "alma", "cuerpo")
analogias("nevera", "mujer", "agua")
analogias("hombre", "carne", "carnival")

"""
Funcion que 
predice las palabras 
mas cercanas
"""

def cercanos(v):

	vecinos = vectores.most_similar(positive = [v])
	print("Vecinos de %s" % v)

	for word, score in vecinos:
		print("\t%s" % word)

cercanos("empatia")
cercanos("vocaciones")
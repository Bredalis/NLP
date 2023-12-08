
"""
Stop Words: son las palabras
mas frecuentes en un texto, 
pero no aportan para 
el analisis del modelo
"""

# Librerias

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Datos para tokenizar

texto = "El gato es negro y el perro es blanco"
texto = texto.lower()

nltk.download('punkt')
stop_words = set(stopwords.words("spanish"))

# Mostrar datos

print("\nStop Words: \n", stop_words)
print("\nCantidad de Stop Words:", len(stop_words))

# Tokenizacion

tokens = word_tokenize(texto)
print("Tokens:", tokens)

# Filtrando tokens sin stop words

texto_filtrado = [word for word in tokens if not word in stop_words]
print("\nTexto sin Stop Words: \n", texto_filtrado)
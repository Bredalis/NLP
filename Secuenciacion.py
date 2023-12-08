
"""
Secuenciación: convertir 
las oraciones en datos
"""

# Librerias

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Datos para tokenizar

sentencias = [
    "Yo amo programar",
    "El NLP es cool",
    "Programar me hace feliz",
    "¡Que viva el NIHILISMO!"
]

# Tokenizacion

tokenizer = Tokenizer(num_words = 100, oov_token = "<desconocido>")
tokenizer.fit_on_texts(sentencias)
word_index = tokenizer.word_index

# Arrays de tokens

secuencia = tokenizer.texts_to_sequences(sentencias)

# Rellenar espacios vacios con ceros

secuencias_acolchadas = pad_sequences(secuencia, padding = "post")

print("Tokens: \n", word_index)
print("Secuencia: \n", secuencia)
print("Secuencia rellenada con ceros: \n", secuencias_acolchadas)

# Prueba

test_data = [
    "El amor esta en el aire",
    "El NIHILISMO es cool"
]

secuencia_de_datos = tokenizer.texts_to_sequences(test_data)

# Rellenar espacios vacios con ceros

secuencias_acolchadas_test = pad_sequences(secuencia_de_datos, maxlen = 10)

print("Secuencia de prueba: \n", secuencia_de_datos)
print("Secuencia rellenada con ceros de prueba: \n", secuencias_acolchadas_test)
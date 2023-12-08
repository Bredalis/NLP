
"""
Tokenizacion: La tokenización es el proceso 
de dividir un texto o una secuencia en unidades 
más pequeñas, llamadas "tokens", representados 
por numeros.
"""

# Libreria

from tensorflow.keras.preprocessing.text import Tokenizer

# Datos para tokenizar

sentencia = [
    "I Love My Dog",
    "I Love My Cat",
    "You Love My Dog!"
]

# Tokenizacion

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentencia)
index_word = tokenizer.index_word

print("Tokens:", index_word)
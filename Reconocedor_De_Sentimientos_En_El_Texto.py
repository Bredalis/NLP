
"""
Capacitar a un modelo
para reconocer el 
sentimiento en el texto
"""

# Librerias

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Variables

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<desconocido>"
training_size = 20000

# Datos

with open("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/Sarcasm/sarcasm.json", "r") as f:
  datastore = json.load(f)

sentencias = []
etiquetas = []
urls = []

for item in datastore:
  sentencias.append(item["headline"])
  etiquetas.append(item["is_sarcastic"])
  urls.append(item["article_link"])

# Division de los datos

training_sentencias = sentencias[0:training_size]
testing_sentencias = sentencias[training_size:]
training_etiquetas = etiquetas[0:training_size]
testing_etiquetas = etiquetas[training_size:]

# Tokenizacion y relleno de los datos

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentencias)
word_index = tokenizer.word_index

training_secuencias = tokenizer.texts_to_sequences(training_sentencias)
training_padded = pad_sequences(training_secuencias, maxlen = max_length, padding = padding_type, truncating = trunc_type)

testing_secuencias = tokenizer.texts_to_sequences(testing_sentencias)
testing_padded = pad_sequences(testing_secuencias, maxlen = max_length, padding = padding_type, truncating = trunc_type)

# Convertir a arreglos

training_padded = np.array(training_padded)
training_etiquetas = np.array(training_etiquetas)
testing_padded = np.array(testing_padded)
testing_etiquetas = np.array(testing_etiquetas)

# Modelo

modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# Compilacion 

modelo.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(modelo.summary())

# Entrenamiento

num_epochs = 30
historial = modelo.fit(
    training_padded, training_etiquetas, epochs = num_epochs, 
    validation_data = (testing_padded, testing_etiquetas), verbose = 2
)

# Dato para prueba

sentencia = [
    "granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"
]

# Predicciones

secuencias = tokenizer.texts_to_sequences(sentencia)
padded = pad_sequences(secuencias, maxlen = max_length, padding = padding_type, truncating = trunc_type)

print(modelo.predict(padded))
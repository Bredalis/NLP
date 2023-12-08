
# Librerias

import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Bidirectional, GlobalMaxPooling1D

# Obtener archivo

with open("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/robert_frost.txt") as story:
  story_data = story.read()

print(f"Datos: \n{story_data}")

def Limpiar_Texto(texto):

  # Borrar caracteres especiales
  texto = re.sub(r'[,"\'()\n“”’.;:-]', '', texto)
  return texto

# Poner los datos en minuscula

lower_data = story_data.lower()
division_de_datos = lower_data.splitlines()

print(division_de_datos)

final = ''

# Limpiar cada linea de los datos

for line in division_de_datos:
  line = Limpiar_Texto(line)
  final += '\n' + line

print(final)

datos_finales = final.split('\n')

print(datos_finales)

# Tokenizacion

tokenizer = Tokenizer(num_words = 1000000)
tokenizer.fit_on_texts(datos_finales)

print(len(tokenizer.word_index))
print(tokenizer.word_index)

# Cantidad de vocabulario

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

secuencia_de_entrada = []

for line in datos_finales:
  token_list = tokenizer.texts_to_sequences([line])[0]

  for i in range(1, len(token_list)):
    n_gram_seq = token_list[: i + 1]
    secuencia_de_entrada.append(n_gram_seq)

print(secuencia_de_entrada)

max_seq_length = max(len(x) for x in secuencia_de_entrada)
print(max_seq_length)

secuencia_de_entrada = np.array(pad_sequences(secuencia_de_entrada, maxlen = max_seq_length, padding = 'pre'))
print(secuencia_de_entrada)

xs, etiquetas = secuencia_de_entrada[:, :-1], secuencia_de_entrada[:, -1]
ys = tf.keras.utils.to_categorical(etiquetas, num_classes = vocab_size)

print(xs)
print(ys)

# Modelo

modelo = Sequential()
modelo.add(Embedding(vocab_size, 124, input_length = max_seq_length - 1))
modelo.add(Dropout(0.2))
modelo.add(LSTM(520, return_sequences = True))
modelo.add(Bidirectional(LSTM(340, return_sequences = True)))
modelo.add(GlobalMaxPooling1D())
modelo.add(Dense(1024, activation = 'relu'))
modelo.add(Dense(vocab_size, activation = 'softmax'))

print(modelo.summary())

# Compilacion

modelo.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

historial = modelo.fit(xs, ys, epochs = 100)

# Función para tomar la entrada del textoo de la semilla del usuario y no de palabras a predecir

def predict_words(seed, no_words):
  for i in range(no_words):
    token_list = tokenizer.texts_to_sequences([seed])[0]
    token_list = pad_sequences([token_list], maxlen = max_seq_length -1, padding = 'pre')
    prediccion = np.argmax(modelo.predict(token_list), axis = 1)

    nueva_palabra = ''

    for word, index in tokenizer.word_index.items():
      if prediccion == index:
        nueva_palabra = word
        break
    seed += " " + nueva_palabra

    print(seed)

# Prueba

seed_texto = 'Two roads diverged'
palabras_generadas = 20

predict_words(seed_texto, palabras_generadas)
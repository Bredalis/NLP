
# Modelos de Markov

"""
Los modelos probabilísticos
de Markov son un tipo de modelo
estadístico que describe la evolución
de sistemas a lo largo del tiempo, donde
la probabilidad de que un evento futuro
depende únicamente del estado presente y
no de estados pasados. Estos modelos se
utilizan para representar y predecir
secuencias de eventos en diversas aplicaciones,
como procesamiento de lenguaje natural, procesamiento
de señales y finanzas.
"""

# Librerias

import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split

# Datos

archivos = [
  "C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/Benedetti.txt",
  "C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/Neruda.txt",
]

textos = []
etiquetas = []

for etiqueta, nombre in enumerate(archivos):
  print(f"{nombre} corresponde a {etiqueta}")

  with open(nombre, "r", encoding = "utf-8") as archivo:
    for line in archivo:
      line = line.rstrip().lower()
      print(line)

      # Eliminacion de signos de puntuacion
      if line:
        line = line.translate(str.maketrans('', '', string.punctuation))
        textos.append(line)
        etiquetas.append(etiqueta)
        print(line)

print("Datos: \n", textos)
print(len(textos))

print("Etiquetas: \n", etiquetas)
print(len(etiquetas))

# Division de datos

x_train, x_test, y_train, y_test = train_test_split(textos, etiquetas, test_size = 0.1, random_state = 42)

print("x train: \n", x_train)
print(len(x_train))

print("x test: \n", x_test)
print(len(x_test))

indice = 1
indice_de_palabra = {"<desc>": 0}

# Construccion de un diccionario de codificacion de palabras o indices

for texto in x_train:
  tokens = texto.split()

  for token in tokens:
    if token not in indice_de_palabra:
      indice_de_palabra[token] = indice
      indice += 1

print(indice_de_palabra)
print(x_train)
print(type(x_train))

# Convertir los datos a enteros

x_train_int = []
x_test_int = []

for texto in x_train:
    tokens = texto.split()
    linea_entero = [indice_de_palabra[token] for token in tokens]
    x_train_int.append(linea_entero)

for texto in x_test:
  tokens = texto.split()
  linea_entero = [indice_de_palabra.get(token, 0) for token in tokens]
  x_test_int.append(linea_entero)

print(x_train_int)
print(x_test_int)

# Matrices de transicion y pobabilidades

V = len(indice_de_palabra)
A0 = np.ones((V, V))
PI0 = np.ones(V)

A1 = np.ones((V, V))
PI1 = np.ones(V)

print("Matriz de Benedetti: \n", A0)
print("Probabilidad de Benedetti: \n", PI0)
print("Matriz de Neruda: \n", A1)
print("Probabilidad de Neruda: \n", PI1)

def Conteos_De_Calcular(text_as_int, A, PI):
  for tokens in text_as_int:
    last_idx = None

    for idx in tokens:
      # Estamos en la primera palabra de la secuencia

      if last_idx == None:
        PI[idx] += 1
      else:
        A[last_idx, idx] += 1

      last_idx = idx

Conteos_De_Calcular([t for t, y in zip(x_train_int, y_train) if y == 0], A0, PI0)
Conteos_De_Calcular([t for t, y in zip(x_test_int, y_train) if y == 1], A1, PI1)

print("Probabilidad de Benedetti: \n", PI0)
print("Matriz de Benedetti: \n", A0)
print("Probabilidad de Neruda: \n", PI1)
print("Matriz de Neruda: \n", A1)

# Convertir a A y PI en matrices de probabilidad valida

A0 /= A0.sum(axis = 1, keepdims = True)
PI0 /= PI0.sum()

A1 /= A1.sum(axis = 1, keepdims = True)
PI1 /= PI1.sum()

print("Probabilidades valida de Neruda: \n", PI1)

# Calcular el logaritmo

A0_log = np.log(A0)
PI0_log = np.log(PI0)

A1_log = np.log(A1)
PI1_log = np.log(PI1)

print("Matriz de Benedetti: \n", A0_log)

print("Matriz de Neruda: \n", A1_log)

count0 = sum(y == 0 for y in y_train)  # Cuenta de etiquetas de clase 0 en y_train
count1 = sum(y == 1 for y in y_train)  # Cuenta de etiquetas de clase 1 en y_train
total = len(y_train)  # Cantidad total de ejemplos de entrenamiento
P0 = count0 / total  # Probabilidad a priori de clase 0
P1 = count1 / total  # Probabilidad a priori de clase 1
logp0 = np.log(P0)  # Logaritmo de la probabilidad a priori de clase 0
logp1 = np.log(P1)  # Logaritmo de la probabilidad a priori de clase 1
print(P0, P1)  # Imprime las probabilidades a priori de ambas clases

# contrucción de un clasificador
class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors) # número de clases

    def _compute_log_likelihood(self, input_, class_):
        logA = self.logAs[class_]
        logpi = self.logpis[class_]

        last_idx = None
        logprob = 0
        for idx in input_:
            if last_idx is None:
                # Es el primer token en la secuencia
                logprob += logpi[idx]
            else:
                # Calcula la probabilidad de transición de la palabra anterior a la actual
                logprob += logA[last_idx, idx]

            # Actualiza last_idx para la próxima iteración
            last_idx = idx

        return logprob

    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            # Calcula los logaritmos de las probabilidades posteriores para cada clase
            posteriors = [self._compute_log_likelihood(input_, c) + self.logpriors[c] \
                          for c in range(self.K)]
            # Elige la clase con la mayor probabilidad posterior como la predicción
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions

# Cada arreglo debe estar en orden ya que se asume que las clases indexan estas listas

clf = Classifier([A0_log, A1_log], [PI0_log, PI1_log], [PI0_log, PI1_log])

Ptrain = clf.predict(x_train_int)
print(f"Train acc: {np.mean(Ptrain == y_train)}")

Ptest = clf.predict(x_test_int)
print(f"Test acc: {np.mean(Ptest == y_test)}")

"""
Spinnnig: es la tecnica
de tomar un articulo ya escrito
y modificarlo (reemplazando palabras,
frases o reorganizando estructuras),
para que parezca un nuevo contenido,
pero manteniendo el mensaje original
"""

# Librerias

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Datos

nltk.download("punkt")

df = pd.read_csv("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/CSV/Base.csv")
print(f"DF\n: {df.head()}")

# Division de datos

textos = df["cuerpo"]
print(textos)
print(f"Primera noticia: {textos[0]}")

probabilidades = {} # key: (w(t-1), w(t+1)), value: {w(t): count(w(t))}

"""
Separar cada noticia
apartir de un punto 
y convertirlas a tokens 
(una lista de palabras)
"""

for doc in textos:
	lineas = doc.split(".")
	for linea in lineas:
		tokens = word_tokenize(linea, language = "spanish")
		print(tokens)

		if len(tokens) >= 2:
			for i in range(len(tokens) - 2):
				t_0 = tokens[i] # Primera palabra
				t_1 = tokens[i + 1] # Palabra del medio
				t_2 = tokens[i + 2] # Ultima palabra
				key = (t_0, t_2)

				if key not in probabilidades:
					probabilidades[key] = {}

				if t_1 not in probabilidades[key]:
					probabilidades[key][t_1] = 1

				else:
					probabilidades[key][t_1] += 1

print(f"Datos divididos por linea: {lineas}")

# Normalizar probabilidades

for key, d in probabilidades.items():
	total = sum(d.values())

	for k, v in d.items():
		d[k] = v / total

print(f"Probabilidades: {probabilidades}")

# Detokenizacion

detokenizador = TreebankWordDetokenizer()

ejemplo = "Hola, Mundo"
token_ejemplo = word_tokenize(ejemplo, language = "spanish")

print(f"Tokens: {token_ejemplo}")
print(f"Tokens detokenizados: {detokenizador.detokenize(token_ejemplo)}")

def Spinnig_Document(doc):
	lineas = doc.split(".")
	output = []

	for linea in lineas:
		if linea:
			new_line = Spinnig_Line(linea)

		else:
			new_line = linea

		output.append(new_line)
	return "\n".join(output)

def Palabra_De_Muestra(d):
	P0 = np.random.random()
	cumulative = 0

	for t, p in d.items():
		cumulative += p

		if P0 < cumulative:
			return t

def Spinnig_Line(linea):
    tokens = word_tokenize(linea, language = 'spanish')
    i = 0
    salida = [tokens[0]]
    if len(tokens) >= 2:
        while i < (len(tokens) - 2):
            t_0 = tokens[i]
            t_1 = tokens[i + 1]
            t_2 = tokens[i + 2]
            key = (t_0, t_2)
            p_dist = probabilidades[key]
            if len(p_dist) > 1 and np.random.random() < 0.3:
                centro = Palabra_De_Muestra(p_dist)
                salida.append(t_1)
                salida.append("<" + centro + ">")
                salida.append(t_2)
                i += 2
            else:
                salida.append(t_1)
                i += 1

            if i == len(tokens) - 2:
              salida.append(tokens[- 1])

            return detokenizador.detokenize(salida)

print(Spinnig_Line("dos días después como estas"))

i = np.random.choice(textos.shape[0])
doc = textos.iloc[i]
new_doc = Spinnig_Document(doc)

print(new_doc)
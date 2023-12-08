
# !pip install gensim -q
# !pip install pypdf2 -q

# Librerias

import string
import PyPDF2
from textwrap import wrap
from gensim.models import Word2Vec, KeyedVectors

def Analogia(v1, v2, v3):
  similitud = embeddings_cargados.most_similar(positive = [v1, v3], negative = [v2])
  print(f"{v1} es a {v2} como {similitud[0][0]} es a {v3}")

# Funcion que extrae datos de pdf

def extraer_texto_desde_pdf(ruta_archivo):
    with open(ruta_archivo, 'rb') as archivo:
        lector = PyPDF2.PdfReader(archivo)
        texto = ""
        for pagina in range(len(lector.pages)):
            texto += lector.pages[pagina].extract_text()
    return texto

documento = extraer_texto_desde_pdf("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/100AñosDeSoledad.pdf")

print("Documento 100 Años de soledad: \n", documento)
print("Longitud de documento 100 años de soledad:", len(documento))

# Dividir el documento en frases usando la coma como separador

oraciones = documento.split(",")

print("Longitude de oraciones:", len(oraciones))
print("Primera oracion: \n", oraciones[0])

oraciones_limpias = []

for oracion in oraciones:
    # Eliminar puntuación y dividir por espacios
    tokens = oracion.translate(str.maketrans('',
                                    '', string.punctuation)).split()
    # Convertir a minúsculas
    tokens = [word.lower() for word in tokens if word.isalpha()]
    if tokens:  # Añadir solo si hay tokens
        oraciones_limpias.append(tokens)

print(oraciones_limpias[0])

# Modelo

modelo = Word2Vec(sentences = oraciones_limpias, window = 5, vector_size = 500, min_count = 1, workers = 8)

def Palabras_Cercanas(palabra):
  palabras_cercanas = modelo.wv.most_similar(palabra, topn = 10)
  print(palabras_cercanas)

Palabras_Cercanas("buendia")

# Guardar Embedding

modelo.wv.save_word2vec_format('100añosdesoledad_emb.txt', binary = False)

embeddings_cargados = KeyedVectors.load_word2vec_format('100añosdesoledad_emb.txt', binary = False)
# O si fue guardado en formato binario:
# embeddings_cargados = KeyedVectors.load_word2vec_format('embeddings.bin', binary=True)
embeddings_cargados

Analogia("rey", "hombre", "mujer")
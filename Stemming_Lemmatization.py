
"""
Stemming: es una tecnica
que elimina los sufijos
de las palabras

Lemmatization: es una tecnica
que utiliza las reglas del
lenguaje para obtener la base
o raiz de una palabra
"""

# !pip install spacy -q
# !python -m spacy download es_core_news_sm -q

# Librerias

import nltk
import spacy
from nltk.stem import SnowballStemmer

nltk.download("wordnet")

# Crear Stemmer en español

stemmer = SnowballStemmer("spanish")

# Prueba del Stemmer

print(stemmer.stem("comiendo"))
print(stemmer.stem("comer"))
print(stemmer.stem("comio"))

# Modelo

nlp = spacy.load("es_core_news_sm")

# Crear documento

doc = nlp("corriendo correr corrio")

# Mostrar texto y el lema de cada token

for token in doc:
  print(token.text, "->", token.lemma_)
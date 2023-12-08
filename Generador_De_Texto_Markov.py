
# Librerias

import numpy as np
import string

# Datos

pa_inicial = {}
primer_orden = {}
segundo_orden = {}

def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))

def add2dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)

with open("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/Neruda.txt", 'r', encoding = 'utf-8') as archivo:
    for line in archivo:
        print(line)

        tokens = remove_punctuation(line.rstrip().lower()).split()
        print(tokens)

        T = len(tokens)
        print(f"Tamaño de la fila: {T}")

        for i in range(T):
            token = tokens[i]
            if i == 0:
                pa_inicial[token] = pa_inicial.get(token, 0.) + 1
                print(f"Palabra inicial: {token}")
            else:
                t_1 = tokens[i-1]
                if i == T - 1:
                    add2dict(segundo_orden, (t_1, token), 'END')
                if i == 1:
                    add2dict(primer_orden, t_1, token)
                else:
                    t_2 = tokens[i-2]
                    add2dict(segundo_orden, (t_2, t_1), token)

print(f"Palabras iniciales: {pa_inicial}")
print(segundo_orden)
print(primer_orden)

# Normalizar

inicial_total = sum(pa_inicial.values())
print(inicial_total)
for t, c in pa_inicial.items():
    pa_inicial[t] = c / inicial_total

pa_inicial

# 'para': ['sobrevivirme', 'que', 'tus', 'que', 'tus', 'qué', 'mi', 'tu']

def list2pdict(ts):
    d = {}  # Crear un diccionario vacío
    n = len(ts)  # Obtener la longitud de la lista de elementos

    # Ciclo para contar la ocurrencia de cada elemento en la lista
    for t in ts:
        d[t] = d.get(t, 0.) + 1

    # Ciclo para convertir los conteos en probabilidades relativas
    for t, c in d.items():
        d[t] = c / n

    return d  # Devolver el diccionario de probabilidades

for t_1, ts in primer_orden.items():
    # replace list with dictionary of probabilities
    primer_orden[t_1] = list2pdict(ts)

print(primer_orden)

for k, ts in segundo_orden.items():
    segundo_orden[k] = list2pdict(ts)

print(segundo_orden)

def palabra_ejemplo(d, imprimir):
    # Genera un número aleatorio en el rango (0, 1)
    p0 = np.random.random()
    if(imprimir == 1):
        print(f"p0: {p0}")

    # Inicializa una variable para realizar la suma acumulativa de probabilidades
    cumulative = 0
    if(imprimir == 1):
        print(f"prob acumulada: {cumulative}")

    # Ciclo que recorre cada clave (t) y su probabilidad (p) en el diccionario (d)
    for t, p in d.items():
        # Agrega la probabilidad actual al valor acumulativo
        cumulative += p
        if(imprimir == 1):
            print(f"item: {t}, Prob; {p}")
            print(f"prob acumulada: {cumulative}")

        # Comprueba si el número aleatorio es menor que la acumulación de probabilidades
        if p0 < cumulative:
            # Si se cumple la condición, devuelve la clave (t) seleccionada
            return t

print(primer_orden["de"])

palabra_ejemplo(primer_orden["de"], 1)

def generador(tamaño):
    for i in range(tamaño):
        oracion = []
        #Palabra Inicial
        pal0 = palabra_ejemplo(pa_inicial, 0)
        oracion.append(pal0)
        #Segunda Palabra
        pal1 = palabra_ejemplo(primer_orden[pal0], 0)
        oracion.append(pal1)

        # Segundo orden hasta el fin
        while True:
            pal2 = palabra_ejemplo(segundo_orden[(pal0, pal1)], 0)
            if pal2 == 'END':
                break
            oracion.append(pal2)
            pal0 = pal1
            pal1 = pal2
        print(' '.join(oracion))

generador(5)
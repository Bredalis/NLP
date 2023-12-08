
# Libreria

import re

# Ejemplo de token

texto = "¡El paisaje es muy bonito hoy!"
texto = texto.lower()

def clean_text(texto):

  # Borrar caracteres especiales
  texto = re.sub(r'[!¡]', '', texto)
  return texto

texto = clean_text(texto)
print("Texto:", texto)

# Separar texto en espacio

token = texto.split()
print("Texto en 'token':", token)
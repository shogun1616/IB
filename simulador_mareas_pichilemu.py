"""
Simulador Matemático de Mareas para la Exploración IB
-----------------------------------------------------

Título:
    "Optimización del Tiempo de Surf en Pichilemu:
     Un Modelo Sinusoidal de Mareas"

Este archivo contiene un simulador completo para modelar mareas usando
funciones sinusoidales, graficarlas y calcular horarios óptimos 
para surfear según criterios definidos por el estudiante.

El código está completamente comentado para documentar:
- Introducción
- Modelado sinusoidal
- Ajuste y validación
- Determinación de horas óptimas
- Reflexión
tal como exige la estructura de una **Exploración Matemática del IB (NS)**.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. DATOS DE EJEMPLO — PUEDES REEMPLAZARLOS POR DATOS DEL SHOA
# -------------------------------------------------------------

# Horas del día (0 a 24)
horas = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])

# Alturas reales de marea para un día (simuladas)
# Ejemplo: datos que podrías obtener del SHOA
alturas = np.array([1.2, 1.8, 1.4, 0.8, 0.6, 0.9, 1.5, 1.9, 1.2])

# -------------------------------------------------------------
# 2. CÁLCULO FORMAL DE PARÁMETROS DEL MODELO SINUSOIDAL
# -------------------------------------------------------------

altura_max = np.max(alturas)
altura_min = np.min(alturas)

# d = línea media
d = (altura_max + altura_min) / 2

# a = amplitud
a = (altura_max - altura_min) / 2

# b = frecuencia (marea semidiurna: 12.42 horas)
periodo = 12.42
b = 2 * np.pi / periodo

# c = fase -> momento del primer máximo
indice_maximo = np.argmax(alturas)
c = horas[indice_maximo]

# -------------------------------------------------------------
# 3. DEFINICIÓN DEL MODELO SINUSOIDAL FINAL
# -------------------------------------------------------------

def h(t):
    """Función sinusoidal del modelo de marea."""
    return a * np.sin(b * (t - c)) + d

# -------------------------------------------------------------
# 4. GRAFICAR DATOS ORIGINALES Y MODELO
# -------------------------------------------------------------

def graficar_modelo():
    """Genera una gráfica profesional del modelo y los datos."""
    t_continuo = np.linspace(0, 24, 500)
    plt.figure(figsize=(10, 5))
    plt.scatter(horas, alturas, label="Datos reales", s=50)
    plt.plot(t_continuo, h(t_continuo), label="Modelo sinusoidal", linewidth=2)
    plt.xlabel("Hora del día")
    plt.ylabel("Altura de la marea (m)")
    plt.title("Modelo Sinusoidal de Mareas - Pichilemu")
    plt.grid(True)
    plt.legend()
    plt.show()

# -------------------------------------------------------------
# 5. CÁLCULO DE HORAS ÓPTIMAS PARA SURFEAR
# -------------------------------------------------------------

def maxima():
    """Calcula los máximos (pleamares) resolviendo h'(t)=0."""
    # derivada: h'(t) = a*b*cos(b(t-c))
    # máx cuando cos(b(t-c)) = 0 → b(t-c)=π/2 + kπ
    k = np.array([0, 1, 2, 3])
    soluciones = (np.pi/2 + k*np.pi)/b + c
    return soluciones[(soluciones >= 0) & (soluciones <= 24)]

def rango_optimo(h_min=1.3, h_max=1.7):
    """Devuelve intervalos donde 1.3 ≤ h(t) ≤ 1.7."""
    t = np.linspace(0, 24, 2000)
    valores = h(t)
    mask = (valores >= h_min) & (valores <= h_max)
    intervalos = []
    inicio = None
    for i in range(len(mask)):
        if mask[i] and inicio is None:
            inicio = t[i]
        if not mask[i] and inicio is not None:
            intervalos.append((inicio, t[i]))
            inicio = None
    return intervalos

# -------------------------------------------------------------
# 6. EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------

def main():
    print("\n=====================================")
    print(" SIMULADOR MATEMÁTICO DE MAREAS (IB) ")
    print("=====================================\n")

    print("Modelo sinusoidal obtenido:\n")
    print(f"  Amplitud (a) = {a:.3f}")
    print(f"  Frecuencia (b) = {b:.3f}")
    print(f"  Fase (c) = {c:.3f} horas")
    print(f"  Línea media (d) = {d:.3f} m\n")

    print("Función modelo:")
    print(f"h(t) = {a:.3f} * sin({b:.3f}(t - {c:.3f})) + {d:.3f}\n")

    print("\n--- HORAS DE PLEAMAR (máximos de marea) ---")
    maximos = maxima()
    for m in maximos:
        print(f"→ Pleamar aprox. a las {m:.2f} horas")

    print("\n--- RANGO ÓPTIMO PARA SURF (1.3 a 1.7 m) ---")
    intervalos = rango_optimo()
    if not intervalos:
        print("No hay intervalos óptimos este día.")
    else:
        for (i, f) in intervalos:
            print(f"→ Desde {i:.2f} h hasta {f:.2f} h")

    print("\nGenerando gráfica del modelo...")
    graficar_modelo()

    print("\nSimulación completada.\n")

# Ejecutar si el archivo se corre directamente
if __name__ == "__main__":
    main()


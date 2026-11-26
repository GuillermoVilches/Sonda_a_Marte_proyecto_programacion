import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
# 1. Datos para calibracion del tubo Pitot ( Voltaje [V] vs Presion [kPa ])
V_sensor = np . array ([0.5 , 1.2 , 2.3 , 3.1 , 4.0 , 4.8])
P_sensor = np . array ([10.1 , 25.5 , 48.0 , 65.2 , 83.9 , 101.3])

# 2. Modelo atmosferico terrestre simplificado ( Altitud [m] vs Densidad [kg/m^])

altitudes_tab = np . array ([0 , 5000 , 10000 , 15000 , 20000 , 25000 , 30000])
densidades_tab = np . array ([1.225 , 0.736 , 0.413 , 0.194 , 0.088 , 0.040 , 0.018])

# 3. Telemetria de vuelo : Tiempo [s] ( muestreo a 1 Hz durante 2 min)
t_vuelo = np . linspace (0 , 120 , 121)

# Perfil de velocidad simulado [m/s] ( modelo logistico )
v_vuelo = 2500 * (1 / (1 + np . exp ( -0.05 * ( t_vuelo - 60) ) ) ) - 118

cond_num = ([0.5 , 1.2 , 2.3 , 3.1 , 4.0 , 4.8])

#matriz del balance de masas planeandolo como Ax=b

A = np.array([
    [1,  1,  1],
    [1,  1, -1],
    [1, -2,  0]
],
 dtype=float)

print("El número de condición de la matriz A es:", np.linalg.cond(A))
b = np.array([50000, 15000, 0], dtype=float)


m = np.linalg.solve(A, b)


m1, m2, m3 = m

print(f"Masa tanque 1 (m1): {m1:.2f} kg")
print(f"Masa tanque 2 (m2): {m2:.2f} kg")
print(f"Masa tanque 3 (m3): {m3:.2f} kg")



# Datos existentes en trabajo.marte.py
V_sensor = np.array([0.5, 1.2, 2.3, 3.1, 4.0, 4.8])
P_sensor = np.array([10.1, 25.5, 48.0, 65.2, 83.9, 101.3])
cond_num = ([0.5, 1.2, 2.3, 3.1, 4.0, 4.8])

# 
# A. REGRESIÓN LINEAL (Ecuación normal: w = (X^T X)^(-1) X^T y)
# 
X = np.column_stack([V_sensor, np.ones_like(V_sensor)])
w = np.linalg.inv(X.T @ X) @ X.T @ P_sensor  # w[0]: pendiente, w[1]: ordenada en el origen

# 
# B. INTERPOLACIÓN POLINÓMICA (Matriz de Vandermonde, grado N - 1 = 5)
# 
grado_interp = len(V_sensor) - 1
A_vand = np.vander(V_sensor)
coef_interp = np.linalg.solve(A_vand, P_sensor)

cond_vand = np.linalg.cond(np.vander(cond_num))

# 
# C. EVALUACIÓN Y GRÁFICA
#
V_fino = np.linspace(V_sensor.min(), V_sensor.max(), 200)
P_reg_fino = np.polyval(w, V_fino)
P_interp_fino = np.polyval(coef_interp, V_fino)

print(f"Coeficientes w: {w}")
print(f"Recta de calibración: P(V) = {w[0]:.4f} * V + ({w[1]:.4f}) [kPa]")
print(f"Número de condición (Vandermonde grado {grado_interp}): {cond_vand:.2e}")

plt.figure(figsize=(8, 5))
plt.scatter(V_sensor, P_sensor, color="black", zorder=5, label="Datos de laboratorio")
plt.plot(V_fino, P_reg_fino, "--", label=f"Regresión lineal (w): $P(V) = {w[0]:.2f}V {w[1]:+.2f}$")
plt.plot(V_fino, P_interp_fino, "-", label=f"Interpolación polinómica (grado {grado_interp})")
plt.xlabel("Señal del sensor $V$ [V]")
plt.ylabel("Presión dinámica $P$ [kPa]")
plt.title("Calibración del Sensor de Presión Dinámica")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 
# FUNCIONES DE CÁLCULO NUMÉRICO (Modularizadas y vectorizadas)
# 

def resolver_balance_masas():
    """
    Apartado 1: Sistema lineal Ax = b usando NumPy (np.linalg.solve).
    Devuelve [m1, m2, m3] en kg y el número de condición de A.
    """
    A_mat = np.array([
        [1,  1,  1],
        [1,  1, -1],
        [1, -2,  0]
    ], dtype=float)
    b_vec = np.array([50000.0, 15000.0, 0.0], dtype=float)
    masas = np.linalg.solve(A_mat, b_vec)
    # Validación física: orden de 10^4 kg por tanque (50 t en total),
    # coherente con la primera etapa de un lanzador pesado hacia Marte.
    return masas, np.linalg.cond(A_mat)


def calibrar_sensor(V, P):
    """
    Apartado 2: Regresión lineal P(V) = theta_1 * V + theta_0 con NumPy (np.polyfit).
    Devuelve (theta_1 [kPa/V], theta_0 [kPa]).
    """
    theta_1, theta_0 = np.polyfit(V, P, 1)
    # Validación física: sensibilidad ~21.11 kPa/V (21.11e3 Pa/V) y offset
    # cercano a cero (-0.28 kPa), confirmando respuesta lineal estable del tubo Pitot.
    return theta_1, theta_0


def calcular_altitud(t, v):
    """
    Apartado 3: Integración 1D acumulada h(t) [m] mediante trapecio vectorizado.
    Función propia: NumPy solo dispone de trapecio total (escalar), no acumulado.
    """
    dt = np.diff(t)
    areas_trapecios = 0.5 * (v[:-1] + v[1:]) * dt
    h = np.concatenate(([0.0], np.cumsum(areas_trapecios)))
    # Validación física: h(120 s) ~ 135 840 m (~135.8 km, orden 10^5 m),
    # superando la línea de Kármán (100 km) tras 2 minutos de ascenso.
    return h


def estimar_densidad(h, alt_tab, dens_tab):
    """
    Apartado 4: Interpolación lineal 1D de la densidad rho(t) [kg/m^3] con np.interp.
    """
    rho = np.interp(h, alt_tab, dens_tab)
    # Validación física: parte de 1.225 kg/m^3 (atmósfera estándar ISA a nivel
    # del mar) y decrece con la altitud. Para h > 30 000 m satura en 0.018 kg/m^3
    # por límite superior de la tabla atmosférica.
    return rho


def calcular_aceleracion(t, v):
    """
    Apartado 5: Derivación numérica 1D a(t) [m/s^2] con np.gradient (edge_order=1).
    Aplica diferencias centrales en puntos interiores y diferencias de 1.er orden
    (adelantada y atrasada) en los extremos, totalmente vectorizado.
    """
    a = np.gradient(v, t, edge_order=1)
    # Validación física: pico máximo ~31.24 m/s^2 (~3.19 g) en t = 60 s,
    # dentro del margen estructural habitual (< 4 g) para proteger la sonda.
    return a


def velocidad_teorica(t):
    """Modelo logístico continuo de velocidad v(t) [m/s]."""
    return 2500.0 / (1.0 + np.exp(-0.05 * (t - 60.0))) - 118.0


def derivada_velocidad(t):
    """Derivada analítica dv/dt [m/s^2] para el método de Newton-Raphson."""
    s = 1.0 / (1.0 + np.exp(-0.05 * (t - 60.0)))
    return 125.0 * s * (1.0 - s)


def encontrar_tiempo_critico(v_target=340.0, t0=60.0, tol=1e-8, max_iter=50):
    """
    Apartado 6: Búsqueda de raíz temporal f(t) = v(t) - v_target = 0 por Newton-Raphson.
    Función propia: NumPy no dispone de un solver de raíces para ecuaciones no polinómicas.

    Justificación del punto inicial (t0 = 60.0 s):
    Corresponde al punto de inflexión de la curva logística, donde la pendiente
    f'(t) es máxima (31.25 m/s^2) y equidista de los extremos [0, 120] s. Así se
    evitan las regiones asintóticas con derivada casi nula y se garantiza
    convergencia cuadrática en todo el rango del Slider [300, 100] """
import numpy as np
import matplotlib.pyplot as plt
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
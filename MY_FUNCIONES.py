"""
Resolución de sistemas de ecuaciones lineales Ax = b

Métodos directos:   Eliminación de Gauss (con pivoteo parcial), Factorización LU (PA = LU)
Métodos iterativos: Jacobi, Gauss-Seidel
"""

import numpy as np


# MÉTODOS DIRECTOS


def gauss(A, b):
    """Eliminación de Gauss con pivoteo parcial y sustitución hacia atrás."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Eliminación hacia adelante
    for k in range(n - 1):
        # Pivoteo parcial: fila con mayor valor absoluto en la columna k
        p = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[p, k]) < 1e-12:
            raise ValueError("La matriz es singular (o casi singular).")
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    if abs(A[n - 1, n - 1]) < 1e-12:
        raise ValueError("La matriz es singular (o casi singular).")

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def lu_descomposicion(A):
    """Devuelve P, L, U tales que P·A = L·U (Doolittle con pivoteo parcial)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for k in range(n - 1):
        p = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[p, k]) < 1e-12:
            raise ValueError("La matriz es singular (o casi singular).")
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    if abs(U[n - 1, n - 1]) < 1e-12:
        raise ValueError("La matriz es singular (o casi singular).")
    return P, L, U


def lu(A, b):
    """Resuelve Ax = b mediante PA = LU: Ly = Pb (adelante) y Ux = y (atrás)."""
    b = np.array(b, dtype=float)
    n = len(b)
    P, L, U = lu_descomposicion(A)
    pb = P @ b

    # Sustitución hacia adelante: L y = P b
    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - L[i, :i] @ y[:i]

    # Sustitución hacia atrás: U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# 
# MÉTODOS ITERATIVOS
# 

def _verificar_iterativo(A, b, x0):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    if np.any(np.abs(np.diag(A)) < 1e-12):
        raise ValueError("Hay ceros en la diagonal; reordena las ecuaciones.")
    return A, b, x


def es_diagonal_dominante(A):
    """Condición suficiente (no necesaria) de convergencia."""
    A = np.abs(np.array(A, dtype=float))
    d = np.diag(A)
    return bool(np.all(d >= A.sum(axis=1) - d))


def jacobi(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Método de Jacobi: x_i^(k+1) = (b_i - sum_{j!=i} a_ij x_j^(k)) / a_ii
    Devuelve (solución, iteraciones, error).
    """
    A, b, x = _verificar_iterativo(A, b, x0)
    D = np.diag(A)
    R = A - np.diagflat(D)

    for k in range(1, max_iter + 1):
        x_nuevo = (b - R @ x) / D
        error = np.linalg.norm(x_nuevo - x, np.inf)
        x = x_nuevo
        if error < tol:
            return x, k, error
    raise RuntimeError(f"Jacobi no convergió en {max_iter} iteraciones "
                       f"(último error: {error:.2e}).")


def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Método de Gauss-Seidel: usa los valores ya actualizados en la misma iteración.
    Devuelve (solución, iteraciones, error).
    """
    A, b, x = _verificar_iterativo(A, b, x0)
    n = len(b)

    for k in range(1, max_iter + 1):
        x_viejo = x.copy()
        for i in range(n):
            s1 = A[i, :i] @ x[:i]            # valores nuevos
            s2 = A[i, i + 1:] @ x_viejo[i + 1:]  # valores de la iteración anterior
            x[i] = (b[i] - s1 - s2) / A[i, i]
        error = np.linalg.norm(x - x_viejo, np.inf)
        if error < tol:
            return x, k, error
    raise RuntimeError(f"Gauss-Seidel no convergió en {max_iter} iteraciones "
                       f"(último error: {error:.2e}).")


# 
# REGRESIÓN LINEAL
# 

def my_regresion_lineal(x, y):
    """
    Calcula los coeficientes de la recta y = theta_1 * x + theta_0
    usando la ecuación normal matricial: theta = (X^T * X)^(-1) * X^T * y
    """
    # Matriz de diseño X: columna de unos para theta_0 y columna con los datos x para theta_1
    X = np.column_stack((np.ones_like(x), x))

    # Resolución vectorizada por mínimos cuadrados
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    theta_0 = theta[0]  # Ordenada en el origen [kPa]
    theta_1 = theta[1]  # Pendiente o sensibilidad [kPa/V]

    return theta_0, theta_1


# 
# DEMOSTRACIÓN
# 

if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)

    A = [[10, -1,  2,  0],
         [-1, 11, -1,  3],
         [ 2, -1, 10, -1],
         [ 0,  3, -1,  8]]
    b = [6, 25, -11, 15]

    print("Sistema Ax = b")
    print("A =\n", np.array(A))
    print("b =", np.array(b))
    print("¿Diagonal dominante? ->", es_diagonal_dominante(A))
    print()

    print("Gauss            :", gauss(A, b))
    print("LU               :", lu(A, b))

    x, it, err = jacobi(A, b)
    print(f"Jacobi           : {x}  ({it} iteraciones, error={err:.2e})")

    x, it, err = gauss_seidel(A, b)
    print(f"Gauss-Seidel     : {x}  ({it} iteraciones, error={err:.2e})")

    print("\nSolución de referencia (numpy):", np.linalg.solve(A, b))

    # Ejemplo de uso con los datos del tubo Pitot:
    print("\n--- Regresión lineal (Tubo Pitot) ---")
    V_sensor = np.array([0.5, 1.2, 2.3, 3.1, 4.0, 4.8])
    P_sensor = np.array([10.1, 25.5, 48.0, 65.2, 83.9, 101.3])

    theta_0, theta_1 = my_regresion_lineal(V_sensor, P_sensor)
    P_ajustada = theta_1 * V_sensor + theta_0

    print("theta_0 (ordenada):", theta_0)
    print("theta_1 (pendiente):", theta_1)
    print("P_ajustada:", P_ajustada)
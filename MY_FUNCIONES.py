"""
Resolución de sistemas de ecuaciones lineales Ax = b, regresión, interpolación e integración

Métodos directos:       Eliminación de Gauss (con pivoteo parcial), Factorización LU (PA = LU)
Métodos iterativos:     Jacobi, Gauss-Seidel
Regresión e interp.:    Regresión lineal, Interpolación polinómica (Vandermonde y Lagrange), Interpolación 1D
Integración numérica:   Fórmulas compuestas de Newton-Cotes (q = 0, 1, 2)
"""

import numpy as np


# ---------------------------------------------------------------------------
# MÉTODOS DIRECTOS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MÉTODOS ITERATIVOS
# ---------------------------------------------------------------------------

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
    """Método de Jacobi. Devuelve (solución, iteraciones, error)."""
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
    """Método de Gauss-Seidel. Devuelve (solución, iteraciones, error)."""
    A, b, x = _verificar_iterativo(A, b, x0)
    n = len(b)

    for k in range(1, max_iter + 1):
        x_viejo = x.copy()
        for i in range(n):
            s1 = A[i, :i] @ x[:i]                 # valores nuevos
            s2 = A[i, i + 1:] @ x_viejo[i + 1:]   # valores de la iteración anterior
            x[i] = (b[i] - s1 - s2) / A[i, i]
        error = np.linalg.norm(x - x_viejo, np.inf)
        if error < tol:
            return x, k, error
    raise RuntimeError(f"Gauss-Seidel no convergió en {max_iter} iteraciones "
                       f"(último error: {error:.2e}).")


# ---------------------------------------------------------------------------
# REGRESIÓN LINEAL E INTERPOLACIÓN
# ---------------------------------------------------------------------------

def vandermonde(x, N, M):
    """Construye la matriz de Vandermonde de dimensión M x N."""
    return np.array([[x[j]**i for i in range(N)] for j in range(M)], dtype=float)


def polynomial(w, x):
    """Evalúa P(x) = sum(w[k] * x**k)."""
    if type(x) == list:
        x = np.array(x, dtype=float)
    N = len(w)
    return sum(w[k] * x**k for k in range(N))


def my_regresion_lineal(x, y):
    """Calcula los coeficientes de la recta y = theta_1 * x + theta_0."""
    # Matriz de diseño X: columna de unos para theta_0 y columna con x para theta_1
    X = np.column_stack((np.ones_like(x), x))

    # Resolución por mínimos cuadrados
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    theta_0 = theta[0]  # Ordenada en el origen
    theta_1 = theta[1]  # Pendiente

    return theta_0, theta_1


def my_interpolacion_polinomica(x, y, xp=None):
    """Interpolación polinómica de grado M - 1 mediante Vandermonde."""
    M = len(x)
    V = vandermonde(x, M, M)
    w = gauss(V, y)

    if xp is None:
        return w
    return polynomial(w, xp)


def my_interpolacion_lagrange(x, y, xp):
    """Interpolación polinómica mediante polinomios base de Lagrange."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if type(xp) == list:
        xp = np.array(xp, dtype=float)

    M = len(x)
    P = 0.0
    for i in range(M):
        L = 1.0
        for j in range(M):
            if j != i:
                L = L * (xp - x[j]) / (x[i] - x[j])
        P = P + y[i] * L
    return P


def my_interpolacion_1D(x, y, xp):
    """Interpolación lineal a tramos en 1D sobre los nodos (x, y)."""
    xs = np.array(x, dtype=float)
    ys = np.array(y, dtype=float)
    M = len(xs)

    # Ordenar los nodos de menor a mayor según xs
    for i in range(M - 1):
        for j in range(i + 1, M):
            if xs[j] < xs[i]:
                (xs[i], xs[j]) = (xs[j], xs[i])
                (ys[i], ys[j]) = (ys[j], ys[i])

    # Si xp es un vector o lista
    if type(xp) == np.ndarray or type(xp) == list:
        yp = np.zeros(len(xp))
        for k, val in enumerate(xp):
            if val <= xs[0]:
                yp[k] = ys[0]
            elif val >= xs[M - 1]:
                yp[k] = ys[M - 1]
            else:
                for i in range(M - 1):
                    if xs[i] <= val <= xs[i + 1]:
                        yp[k] = ys[i] + (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) * (val - xs[i])
                        break
        return yp

    # Si xp es un escalar
    else:
        if xp <= xs[0]:
            return float(ys[0])
        elif xp >= xs[M - 1]:
            return float(ys[M - 1])
        else:
            for i in range(M - 1):
                if xs[i] <= xp <= xs[i + 1]:
                    return float(ys[i] + (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) * (xp - xs[i]))


# ---------------------------------------------------------------------------
# INTEGRACIÓN NUMÉRICA (q = 0, 1, 2)
# ---------------------------------------------------------------------------

def my_integracion_numerica(f, a, b, N=1000, q=1):
    """Aproxima la integral de f en [a, b] con N subintervalos para q = 0, 1, 2."""
    h = (b - a) / N
    y = np.array([f(a + h * i) for i in range(N + 1)], dtype=float)

    # q = 0: fórmula de los rectángulos
    if q == 0:
        return h * np.sum(y[:N])

    # q = 1: fórmula del trapecio
    elif q == 1:
        return h * (0.5 * y[0] + np.sum(y[1:N]) + 0.5 * y[N])

    # q = 2: fórmula de Simpson (requiere N par)
    elif q == 2:
        if N % 2 != 0:
            raise ValueError("Para q = 2 (Simpson), N debe ser un número par.")
        return (h / 3.0) * (y[0] + 4.0 * np.sum(y[1:N:2]) + 2.0 * np.sum(y[2:N:2]) + y[N])

    else:
        raise ValueError("El grado q debe ser 0, 1 o 2.")


# ---------------------------------------------------------------------------
# VERIFICACIÓN DE TODAS LAS FUNCIONES
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)
    tol = 1e-7

    # 1. Sistemas lineales (Gauss, LU, Jacobi, Gauss-Seidel)
    print("--- 1. Sistemas lineales ---")
    A = np.array([[10, -1,  2,  0],
                  [-1, 11, -1,  3],
                  [ 2, -1, 10, -1],
                  [ 0,  3, -1,  8]], dtype=float)
    b = np.array([6, 25, -11, 15], dtype=float)

    x_ref = np.linalg.solve(A, b)
    x_gauss = gauss(A, b)
    x_lu = lu(A, b)
    x_jac, it_j, _ = jacobi(A, b)
    x_gs, it_gs, _ = gauss_seidel(A, b)

    print("Error Gauss        :", np.max(np.abs(x_gauss - x_ref)))
    print("Error LU           :", np.max(np.abs(x_lu - x_ref)))
    print(f"Error Jacobi       : {np.max(np.abs(x_jac - x_ref)):.2e} ({it_j} iter)")
    print(f"Error Gauss-Seidel : {np.max(np.abs(x_gs - x_ref)):.2e} ({it_gs} iter)")

    assert np.max(np.abs(x_gauss - x_ref)) < tol
    assert np.max(np.abs(x_lu - x_ref)) < tol
    assert np.max(np.abs(x_jac - x_ref)) < tol
    assert np.max(np.abs(x_gs - x_ref)) < tol

    # 2. Regresión lineal
    print("\n--- 2. Regresión lineal ---")
    V_sensor = np.array([0.5, 1.2, 2.3, 3.1, 4.0, 4.8])
    P_sensor = np.array([10.1, 25.5, 48.0, 65.2, 83.9, 101.3])

    theta_0, theta_1 = my_regresion_lineal(V_sensor, P_sensor)
    m_ref, n_ref = np.polyfit(V_sensor, P_sensor, 1)

    print("Error theta_0 :", abs(theta_0 - n_ref))
    print("Error theta_1 :", abs(theta_1 - m_ref))

    assert abs(theta_0 - n_ref) < tol
    assert abs(theta_1 - m_ref) < tol

    # 3. Interpolación polinómica (Vandermonde y Lagrange)
    print("\n--- 3. Interpolación polinómica y Lagrange ---")
    x_nodos = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y_nodos = np.array([1.0, 2.0, 0.0, 5.0, 4.0, 3.0])
    xp = np.linspace(0.0, 5.0, 50)

    yp_vand = my_interpolacion_polinomica(x_nodos, y_nodos, xp)
    yp_lagr = my_interpolacion_lagrange(x_nodos, y_nodos, xp)
    yp_ref = np.polyval(np.polyfit(x_nodos, y_nodos, len(x_nodos) - 1), xp)

    print("Error nodos Vandermonde :", np.max(np.abs(my_interpolacion_polinomica(x_nodos, y_nodos, x_nodos) - y_nodos)))
    print("Error nodos Lagrange    :", np.max(np.abs(my_interpolacion_lagrange(x_nodos, y_nodos, x_nodos) - y_nodos)))
    print("Error Vandermonde-Ref   :", np.max(np.abs(yp_vand - yp_ref)))
    print("Error Lagrange-Ref      :", np.max(np.abs(yp_lagr - yp_ref)))

    assert np.max(np.abs(yp_vand - yp_ref)) < tol
    assert np.max(np.abs(yp_lagr - yp_ref)) < tol

    # 4. Interpolación 1D (con nodos desordenados y puntos fuera de rango)
    print("\n--- 4. Interpolación 1D ---")
    x_desord = np.array([3.0, 0.0, 5.0, 1.0, 4.0, 2.0])
    y_desord = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 0.0])
    xp_1d = np.array([-1.0, 0.5, 2.5, 4.5, 6.0])

    yp_1d = my_interpolacion_1D(x_desord, y_desord, xp_1d)
    yp_1d_ref = np.interp(xp_1d, x_nodos, y_nodos)
    yp_esc = my_interpolacion_1D(x_desord, y_desord, 2.5)

    print("Error vector vs np.interp  :", np.max(np.abs(yp_1d - yp_1d_ref)))
    print("Error escalar vs np.interp :", abs(yp_esc - np.interp(2.5, x_nodos, y_nodos)))

    assert np.max(np.abs(yp_1d - yp_1d_ref)) < tol
    assert abs(yp_esc - np.interp(2.5, x_nodos, y_nodos)) < tol

    # 5. Integración numérica (q = 0, 1, 2)
    print("\n--- 5. Integración numérica ---")
    f_cuad = lambda x: 3.0 * x**2 + 2.0 * x + 1.0
    err_simpson = abs(my_integracion_numerica(f_cuad, 0.0, 2.0, N=2, q=2) - 14.0)
    print("Error Simpson en parábola (N=2):", err_simpson)
    assert err_simpson < tol

    err_q0 = abs(my_integracion_numerica(np.sin, 0.0, np.pi / 2, N=100, q=0) - 1.0)
    err_q1 = abs(my_integracion_numerica(np.sin, 0.0, np.pi / 2, N=100, q=1) - 1.0)
    err_q2 = abs(my_integracion_numerica(np.sin, 0.0, np.pi / 2, N=100, q=2) - 1.0)
    print(f"sin(x) [0, pi/2] -> Error q=0: {err_q0:.2e} | q=1: {err_q1:.2e} | q=2: {err_q2:.2e}")

    assert err_q2 < err_q1 < err_q0
    assert err_q2 < 1e-8

    print("\nTODAS LAS VERIFICACIONES OK")
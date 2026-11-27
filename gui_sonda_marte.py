# 
# gui_sonda_marte.py - Interfaz Gráfica Interactiva (Dashboard Ares-I)
# 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider


def lanzar_dashboard(
    V_sensor,
    P_sensor,
    t_vuelo,
    masas,
    cond_A,
    theta_1,
    theta_0,
    h_vuelo,
    rho_vuelo,
    a_vuelo,
    velocidad_teorica,
    encontrar_tiempo_critico,
):
    """
    Orquesta el dashboard interactivo único con RadioButtons para los 6 apartados,
    Slider dinámico exclusivo del Apartado 6 y displays numéricos en unidades SI.
    """
    # Ventana principal y reserva de espacio para widgets
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            "Dashboard Telemetría Ares-I - Misión Marte"
        )
    plt.subplots_adjust(left=0.30, bottom=0.20, right=0.95, top=0.90)

    opciones = [
        "1. Balance de Masas",
        "2. Calibración Pitot",
        "3. Trayectoria h(t)",
        "4. Densidad rho(t)",
        "5. Aceleración a(t)",
        "6. Instante Crítico",
    ]

    ax_radio = plt.axes([0.02, 0.34, 0.23, 0.48], facecolor="#f4f6f8")
    radio = RadioButtons(ax_radio, opciones, active=0)

    ax_slider = plt.axes([0.30, 0.06, 0.65, 0.04], facecolor="#e8eef5")
    slider_v = Slider(
        ax_slider,
        "v_target [m/s]",
        300.0,
        1000.0,
        valinit=340.0,
        valstep=1.0,
    )

    estilo_caja = dict(
        boxstyle="round,pad=0.5",
        facecolor="#f8f9fa",
        edgecolor="#2c3e50",
        alpha=0.92,
    )

    def dibujar_apartado_6(v_target):
        ax.clear()
        t_crit, iters = encontrar_tiempo_critico(v_target)
        t_fino = np.linspace(0.0, 120.0, 400)
        v_fino = velocidad_teorica(t_fino)

        ax.plot(
            t_fino, v_fino, color="#1f77b4", lw=2, label="Velocidad teórica $v(t)$"
        )
        ax.axhline(
            v_target,
            color="#7f8c8d",
            ls="--",
            lw=1.2,
            label=f"Velocidad diana $v_{{target}} = {v_target:.0f}$ m/s",
        )
        ax.axvline(t_crit, color="#d62728", ls=":", lw=1.2)
        ax.scatter(
            [t_crit],
            [v_target],
            color="#d62728",
            s=70,
            zorder=5,
            label=f"Raíz: $t_{{crit}} = {t_crit:.2f}$ s",
        )

        ax.set_title("Apartado 6: Instante Crítico de Vuelo (Newton-Raphson)")
        ax.set_xlabel("Tiempo de vuelo $t$ [s]")
        ax.set_ylabel("Velocidad $v(t)$ [m/s]")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")

        texto = (
            f"v_target = {v_target:.1f} m/s  |  t_crit = {t_crit:.2f} s ({iters} iter., t0 = 60 s)\n"
            "Significado físico: instante exacto en que el Ares-I alcanza la velocidad\n"
            "objetivo (340 m/s ≈ Mach 1, entrada en régimen supersónico)."
        )
        ax.text(
            0.03,
            0.95,
            texto,
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
            bbox=estilo_caja,
        )
        fig.canvas.draw_idle()

    def actualizar_vista(modo):
        ax.clear()
        es_apartado_6 = modo == opciones[5]
        ax_slider.set_visible(es_apartado_6)
        slider_v.set_active(es_apartado_6)

        if modo == opciones[0]:
            tanques = ["Tanque 1 ($m_1$)", "Tanque 2 ($m_2$)", "Tanque 3 ($m_3$)"]
            ax.bar(
                tanques,
                masas,
                color=["#1f77b4", "#2ca02c", "#ff7f0e"],
                edgecolor="black",
                label="Masa de propelente [kg]",
            )
            ax.set_ylim(0, float(np.max(masas)) * 1.35)
            ax.set_title(
                r"Apartado 1: Balance de Masas de Propelente ($A\vec{x}=\vec{b}$)"
            )
            ax.set_xlabel("Depósitos de la primera etapa")
            ax.set_ylabel("Masa [kg]")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.legend(loc="upper right")
            texto = (
                f"m1 = {masas[0]:.2f} kg | m2 = {masas[1]:.2f} kg | m3 = {masas[2]:.2f} kg (cond(A) = {cond_A:.2f})\n"
                "Significado físico: reparto óptimo de 50 000 kg de propelente que asegura\n"
                "la estabilidad aerodinámica y los flujos másicos del lanzador."
            )
            ax.text(
                0.03,
                0.95,
                texto,
                transform=ax.transAxes,
                va="top",
                fontsize=9.5,
                bbox=estilo_caja,
            )

        elif modo == opciones[1]:
            V_grid = np.linspace(V_sensor.min(), V_sensor.max(), 200)
            P_fit = theta_1 * V_grid + theta_0
            ax.scatter(
                V_sensor, P_sensor, color="black", zorder=5, label="Datos de laboratorio"
            )
            ax.plot(
                V_grid,
                P_fit,
                "--",
                color="#d62728",
                lw=2,
                label=f"Ajuste lineal: $P(V) = {theta_1:.2f}V {theta_0:+.2f}$",
            )
            ax.set_title("Apartado 2: Calibración del Sensor de Presión Dinámica")
            ax.set_xlabel("Voltaje del transductor $V$ [V]")
            ax.set_ylabel("Presión dinámica $P$ [kPa]")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="lower right")
            texto = (
                f"Pendiente θ1 = {theta_1:.2f} kPa/V ({theta_1 * 1e3:.0f} Pa/V) | Offset θ0 = {theta_0:.2f} kPa\n"
                "Significado físico: relación lineal con error de cero despreciable para\n"
                "monitorizar la presión dinámica durante el ascenso atmosférico."
            )
            ax.text(
                0.03,
                0.95,
                texto,
                transform=ax.transAxes,
                va="top",
                fontsize=9.5,
                bbox=estilo_caja,
            )

        elif modo == opciones[2]:
            ax.plot(
                t_vuelo,
                h_vuelo,
                color="#1f77b4",
                lw=2,
                label="Altitud reconstruida $h(t)$",
            )
            ax.axhline(
                100000.0,
                color="#7f8c8d",
                ls="--",
                lw=1.1,
                label="Línea de Kármán (100 km)",
            )
            ax.set_title(
                "Apartado 3: Reconstrucción de la Trayectoria (Trapecio Acumulado)"
            )
            ax.set_xlabel("Tiempo de vuelo $t$ [s]")
            ax.set_ylabel("Altitud $h(t)$ [m]")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="lower right")
            texto = (
                f"Altitud final h(120 s) = {h_vuelo[-1]:.1f} m ({h_vuelo[-1] / 1000:.2f} km)\n"
                "Significado físico: reconstrucción posicional sin radar que confirma\n"
                "la salida de la atmósfera densa (> 100 km) antes de los 2 minutos."
            )
            ax.text(
                0.03,
                0.95,
                texto,
                transform=ax.transAxes,
                va="top",
                fontsize=9.5,
                bbox=estilo_caja,
            )

        elif modo == opciones[3]:
            ax.plot(
                t_vuelo,
                rho_vuelo,
                color="#2ca02c",
                lw=2,
                label=r"Densidad estimada $\rho(t)$",
            )
            ax.set_ylim(-0.05, float(np.max(rho_vuelo)) * 1.15)
            ax.set_title(
                "Apartado 4: Estimación de la Densidad Atmosférica (Interpolación 1D)"
            )
            ax.set_xlabel("Tiempo de vuelo $t$ [s]")
            ax.set_ylabel(r"Densidad atmosférica $\rho$ [kg/m$^3$]")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right")
            texto = (
                f"rho(0 s) = {rho_vuelo[0]:.3f} kg/m^3 | rho(120 s) = {rho_vuelo[-1]:.3f} kg/m^3\n"
                "Significado físico: la densidad decae bruscamente al ganar altitud,\n"
                "disminuyendo las cargas y la resistencia aerodinámica (satura > 30 km)."
            )
            ax.text(
                0.97,
                0.76,
                texto,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.5,
                bbox=estilo_caja,
            )

        elif modo == opciones[4]:
            idx_max = int(np.argmax(a_vuelo))
            ax.plot(
                t_vuelo,
                a_vuelo,
                color="#ff7f0e",
                lw=2,
                label="Aceleración axial $a(t)$",
            )
            ax.scatter(
                [t_vuelo[idx_max]],
                [a_vuelo[idx_max]],
                color="#d62728",
                s=60,
                zorder=5,
                label=f"Máximo: {a_vuelo[idx_max]:.2f} m/s$^2$ en $t = {t_vuelo[idx_max]:.0f}$ s",
            )
            ax.set_ylim(0, float(np.max(a_vuelo)) * 1.35)
            ax.set_title("Apartado 5: Aceleración Estructural (Diferencias Finitas)")
            ax.set_xlabel("Tiempo de vuelo $t$ [s]")
            ax.set_ylabel("Aceleración axial $a(t)$ [m/s$^2$]")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="lower center")
            texto = (
                f"a_max = {a_vuelo[idx_max]:.2f} m/s^2 ({a_vuelo[idx_max] / 9.81:.2f} g) en t = {t_vuelo[idx_max]:.0f} s\n"
                "Significado físico: carga inercial máxima sobre la estructura y la sonda\n"
                "científica; permanece por debajo del límite estructural seguro (~4 g)."
            )
            ax.text(
                0.03,
                0.95,
                texto,
                transform=ax.transAxes,
                va="top",
                fontsize=9.5,
                bbox=estilo_caja,
            )

        elif modo == opciones[5]:
            dibujar_apartado_6(slider_v.val)
            return

        fig.canvas.draw_idle()

    radio.on_clicked(actualizar_vista)
    slider_v.on_changed(
        lambda val: dibujar_apartado_6(val)
        if radio.value_selected == opciones[5]
        else None
    )

    actualizar_vista(opciones[0])
    plt.show()


if __name__ == "__main__":
    # Permite ejecutar directamente este archivo si Sonda_marte.py está en la misma carpeta
    import Sonda_marte as sm

    masas, cond_A = sm.resolver_balance_masas()
    theta_1, theta_0 = sm.calibrar_sensor(sm.V_sensor, sm.P_sensor)
    h_vuelo = sm.calcular_altitud(sm.t_vuelo, sm.v_vuelo)
    rho_vuelo = sm.estimar_densidad(h_vuelo, sm.altitudes_tab, sm.densidades_tab)
    a_vuelo = sm.calcular_aceleracion(sm.t_vuelo, sm.v_vuelo)

    lanzar_dashboard(
        V_sensor=sm.V_sensor,
        P_sensor=sm.P_sensor,
        t_vuelo=sm.t_vuelo,
        masas=masas,
        cond_A=cond_A,
        theta_1=theta_1,
        theta_0=theta_0,
        h_vuelo=h_vuelo,
        rho_vuelo=rho_vuelo,
        a_vuelo=a_vuelo,
        velocidad_teorica=sm.velocidad_teorica,
        encontrar_tiempo_critico=sm.encontrar_tiempo_critico,
    )

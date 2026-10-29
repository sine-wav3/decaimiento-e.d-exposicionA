import numpy as np
import matplotlib

try:
    _en_notebook = get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # Jupyter/Colab
except NameError:
    _en_notebook = False

if not _en_notebook:
    import matplotlib.pyplot as _plt
    for _b in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX", "WXAgg"):
        try:
            matplotlib.use(_b, force=True)
            _plt.figure(); _plt.close("all")   # falla si ese backend no está disponible
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

_bk = matplotlib.get_backend().lower()
if any(x in _bk for x in ("agg", "inline", "svg", "pdf")) and "tk" not in _bk and "qt" not in _bk:
    print(f"AVISO: el backend actual es '{_bk}' y NO es interactivo: los botones no responderán.\n"
          "  - Script normal: pip install pyqt5   (o instala tkinter)\n"
          "  - Jupyter: usa  %matplotlib qt   o   %matplotlib widget  (requiere: pip install ipympl)\n"
          "  - Spyder: Preferencias > Consola IPython > Gráficos > Backend: Automático o Qt")

# Valores del ejemplo de Zill (también los usa el botón "Reiniciar")
ZILL = dict(T0=300.0, Tm=70.0, T1=200.0, t1=3.0, tmax=40.0, delta=5.0)

# Tabla de la figura 2.37b del libro: (T en °F, t en min)
TABLA_ZILL = [(75, 20.1), (74, 21.3), (73, 22.8), (72, 24.9), (71, 28.6), (70.5, 32.3)]

estado = dict(animando=False, t_anim=0.0, campo=False, tabla=True, tau=True)

# ---------------------------------------------------------------- figura
fig = plt.figure(figsize=(14, 8.5))
try:
    fig.canvas.manager.set_window_title("Ley de enfriamiento de Newton")
except Exception:
    pass
ax = fig.add_axes([0.07, 0.40, 0.62, 0.54])
ax_info = fig.add_axes([0.71, 0.47, 0.28, 0.47])
ax_info.axis("off")


def crear_slider(y, etiqueta, vmin, vmax, vinit, paso, fmt="%.1f"):
    s_ax = fig.add_axes([0.14, y, 0.50, 0.028])
    return Slider(s_ax, etiqueta, vmin, vmax, valinit=vinit, valstep=paso, valfmt=fmt)


s_T0 = crear_slider(0.30, "T₀ (°F)", -50, 500, ZILL["T0"], 1)
s_Tm = crear_slider(0.255, "Tₘ (°F)", -50, 200, ZILL["Tm"], 1)
s_T1 = crear_slider(0.21, "T(t₁) (°F)", -50, 500, ZILL["T1"], 1)
s_t1 = crear_slider(0.165, "t₁ (min)", 0.5, 15, ZILL["t1"], 0.5)
s_tmax = crear_slider(0.12, "Tiempo mostrado (min)", 10, 120, ZILL["tmax"], 5, "%d")
s_delta = crear_slider(0.075, "Margen |T−Tₘ| (°F)", 0.1, 30, ZILL["delta"], 0.1)
SLIDERS = [s_T0, s_Tm, s_T1, s_t1, s_tmax, s_delta]
CLAVES = ["T0", "Tm", "T1", "t1", "tmax", "delta"]


def crear_boton(y, texto):
    b_ax = fig.add_axes([0.74, y, 0.23, 0.05])
    return Button(b_ax, texto, hovercolor="0.85")


b_anim = crear_boton(0.33, "Animar enfriamiento")
b_campo = crear_boton(0.265, "Campo de direcciones: OFF")
b_tabla = crear_boton(0.20, "Datos del libro (fig. 2.37b): ON")
b_tau = crear_boton(0.135, "Constante de tiempo τ: ON")
b_reset = crear_boton(0.07, "Reiniciar (ejemplo de Zill)")

# ---------------------------------------------------------------- modelo
def parametros():
    T0, Tm, T1, t1 = s_T0.val, s_Tm.val, s_T1.val, s_t1.val
    d0, d1 = T0 - Tm, T1 - Tm
    if d0 == 0 or d1 / d0 <= 0 or d1 / d0 >= 1:
        k = None
    else:
        k = np.log(d1 / d0) / t1
    return T0, Tm, T1, t1, s_tmax.val, s_delta.val, k


def T_de_t(t, T0, Tm, k):
    return Tm + (T0 - Tm) * np.exp(k * t)


marcador = {"punto": None, "texto": None}


def dibujar_todo(_=None):
    T0, Tm, T1, t1, tmax, delta, k = parametros()
    ax.clear()
    ax_info.clear()
    ax_info.axis("off")
    marcador["punto"] = marcador["texto"] = None

    ax.set_xlabel("Tiempo t (min)", fontsize=11)
    ax.set_ylabel("Temperatura T (°F)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, tmax)
    pad = 0.08 * (abs(T0 - Tm) + 20)
    y_lo, y_hi = min(T0, Tm, T1) - pad, max(T0, Tm, T1) + pad
    ax.set_ylim(y_lo, y_hi)
    ax.axhline(Tm, color="tab:green", ls="--", lw=1.6)
    ax.text(tmax * 0.99, Tm, f"Tₘ = {Tm:g} °F ", color="tab:green",
            ha="right", va="bottom", fontsize=10, fontweight="bold")

    if k is None:
        ax.set_title("Datos no válidos: T(t₁) debe estar entre Tₘ y T₀", color="crimson", fontsize=13)
        ax.plot([0], [T0], "ko")
        ax_info.text(0, 0.95, "Ajusta los deslizadores:\n\nT(t₁) debe quedar\nentre T₀ y Tₘ,\ny T₀ ≠ Tₘ.",
                     fontsize=12, color="crimson", va="top")
        fig.canvas.draw_idle()
        return

    ax.set_title("Ley de enfriamiento de Newton  —  Zill, Ejemplo 4", fontsize=13)

    if estado["campo"]:
        sx, sy = tmax, (y_hi - y_lo)
        largo = 0.025
        for ti in np.linspace(0, tmax, 24):
            for Tj in np.linspace(y_lo, y_hi, 18):
                v = np.array([1 / sx, k * (Tj - Tm) / sy])
                v = v / np.hypot(*v) * largo
                ax.plot([ti - v[0] * sx / 2, ti + v[0] * sx / 2],
                        [Tj - v[1] * sy / 2, Tj + v[1] * sy / 2],
                        color="0.65", lw=0.9, zorder=1)

    t = np.linspace(0, tmax, 600)
    ax.plot(t, T_de_t(t, T0, Tm, k), color="tab:blue", lw=2.8, label="T(t)", zorder=3)

    ax.plot([0], [T0], "o", color="tab:red", ms=9, zorder=5)
    ax.annotate(f"T(0) = {T0:g}", (0, T0), textcoords="offset points", xytext=(10, 8),
                fontsize=10, color="tab:red")
    if t1 <= tmax:
        ax.plot([t1], [T1], "s", color="tab:orange", ms=9, zorder=5)
        ax.annotate(f"T({t1:g}) = {T1:g}", (t1, T1), textcoords="offset points",
                    xytext=(10, 10), fontsize=10, color="tab:orange")

    tau = 1 / abs(k)
    if estado["tau"] and tau <= tmax:
        Tt = T_de_t(tau, T0, Tm, k)
        ax.vlines(tau, y_lo, Tt, color="purple", ls=":", lw=1.5)
        ax.plot([tau], [Tt], "D", color="purple", ms=7, zorder=5)
        ax.annotate(f"t = τ = {tau:.2f} min\n(T−Tₘ cae a 37 %)", (tau, Tt),
                    textcoords="offset points", xytext=(12, 12), fontsize=9, color="purple")

    if estado["tabla"]:
        ax.plot([p[1] for p in TABLA_ZILL], [p[0] for p in TABLA_ZILL], "k^", ms=6,
                label="Tabla del libro (fig. 2.37b)", zorder=4)

    signo = 1 if T0 > Tm else -1
    T_obj = Tm + signo * delta
    if delta < abs(T0 - Tm):
        t_obj = np.log(delta / abs(T0 - Tm)) / k
        ax.hlines(T_obj, 0, min(t_obj, tmax), color="tab:red", ls="--", lw=1.3)
        if t_obj <= tmax:
            ax.vlines(t_obj, y_lo, T_obj, color="tab:red", ls="--", lw=1.3)
            ax.plot([t_obj], [T_obj], "o", color="tab:red", ms=7, zorder=5)
        txt_obj = f"T = {T_obj:g} °F se alcanza\nen t = {t_obj:.2f} min"
    else:
        txt_obj = "El margen es mayor que |T₀ − Tₘ|:\nya se cumple en t = 0"

    if estado["animando"] or estado["t_anim"] > 0:
        ta = estado["t_anim"]
        marcador["punto"], = ax.plot([ta], [T_de_t(ta, T0, Tm, k)], "o", color="black",
                                     ms=12, zorder=6)
        marcador["texto"] = ax.text(0.02, 0.05, "", transform=ax.transAxes, fontsize=12,
                                    bbox=dict(boxstyle="round", fc="lightyellow", ec="0.5"))
        actualizar_marcador()

    ax.legend(loc="upper right", fontsize=9)

    signo_txt = "+" if T0 - Tm >= 0 else "-"
    info = (
        r"$\frac{dT}{dt}=k\,(T-T_m)$" "\n\n"
        + rf"$k=\frac{{1}}{{{t1:g}}}\ln\frac{{{T1 - Tm:g}}}{{{T0 - Tm:g}}}={k:.5f}$ min$^{{-1}}$" "\n\n"
        + rf"$T(t)={Tm:g}\,{signo_txt}\,{abs(T0 - Tm):g}\,e^{{{k:.5f}\,t}}$" "\n\n"
        + f"τ = 1/|k| = {tau:.2f} min\n"
        + f"Vida media de (T−Tₘ) = {np.log(2) / abs(k):.2f} min\n\n"
        + txt_obj
    )
    ax_info.text(0, 1, info, va="top", fontsize=12,
                 bbox=dict(boxstyle="round", fc="#f4f8ff", ec="0.7"))
    fig.canvas.draw_idle()


# ------------------------------------------------------------- animación
def actualizar_marcador():
    T0, Tm, T1, t1, tmax, delta, k = parametros()
    if k is None or marcador["punto"] is None:
        return
    ta = estado["t_anim"]
    Ta = T_de_t(ta, T0, Tm, k)
    marcador["punto"].set_data([ta], [Ta])
    marcador["texto"].set_text(f"t = {ta:.1f} min\nT = {Ta:.2f} °F\nT − Tₘ = {Ta - Tm:.2f} °F")


def detener_animacion():
    estado["animando"] = False
    timer.stop()
    b_anim.label.set_text("Animar enfriamiento")


def paso_animacion():
    if not estado["animando"]:
        return
    tmax = s_tmax.val
    estado["t_anim"] = min(estado["t_anim"] + tmax / 150, tmax)
    if estado["t_anim"] >= tmax:
        detener_animacion()
    if marcador["punto"] is None:
        dibujar_todo()
    else:
        actualizar_marcador()
        fig.canvas.draw_idle()


timer = fig.canvas.new_timer(interval=40)
timer.add_callback(paso_animacion)


def alternar_animacion(_):
    if estado["animando"]:
        detener_animacion()
    else:
        if estado["t_anim"] >= s_tmax.val:
            estado["t_anim"] = 0.0
        estado["animando"] = True
        b_anim.label.set_text("Pausar animación")
        timer.start()
        dibujar_todo()
    fig.canvas.draw_idle()


# --------------------------------------------------------------- botones
def alternar_campo(_):
    estado["campo"] = not estado["campo"]
    b_campo.label.set_text(f"Campo de direcciones: {'ON' if estado['campo'] else 'OFF'}")
    dibujar_todo()


def alternar_tabla(_):
    estado["tabla"] = not estado["tabla"]
    b_tabla.label.set_text(f"Datos del libro (fig. 2.37b): {'ON' if estado['tabla'] else 'OFF'}")
    dibujar_todo()


def alternar_tau(_):
    estado["tau"] = not estado["tau"]
    b_tau.label.set_text(f"Constante de tiempo τ: {'ON' if estado['tau'] else 'OFF'}")
    dibujar_todo()


def reiniciar(_):
    detener_animacion()
    estado["t_anim"] = 0.0
    for s, clave in zip(SLIDERS, CLAVES):
        s.eventson = False
        s.set_val(ZILL[clave])
        s.eventson = True
    dibujar_todo()



#def guardar(_):
#   fig.savefig("enfriamiento_newton.png", dpi=200, bbox_inches="tight")
#    print("Imagen guardada como enfriamiento_newton.png")


b_anim.on_clicked(alternar_animacion)
b_campo.on_clicked(alternar_campo)
b_tabla.on_clicked(alternar_tabla)
b_tau.on_clicked(alternar_tau)
b_reset.on_clicked(reiniciar)
#b_png.on_clicked(guardar)
for s in SLIDERS:
    s.on_changed(dibujar_todo)

dibujar_todo()

if __name__ == "__main__":
    plt.show()
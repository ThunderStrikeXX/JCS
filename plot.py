import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Caricamento Dati
try:
    df = pd.read_csv("history.csv")
except FileNotFoundError:
    print("Errore: File 'history.csv' non trovato.")
    exit()

times = df['time'].unique()
n_steps = len(times)

# 2. Setup Figura (Griglia 2x2)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3)

# Flatten degli assi per accedervi facilmente come lista [0,1,2,3]
ax_rho = axs[0, 0]
ax_u   = axs[0, 1]
ax_p   = axs[1, 0]
ax_T   = axs[1, 1]

# 3. Calcolo Limiti Globali (per assi stabili)
def add_margin(series):
    mn, mx = series.min(), series.max()
    rng = mx - mn
    if rng == 0: rng = 0.1
    return mn - 0.1 * rng, mx + 0.1 * rng

ylims = {
    'rho': add_margin(df['rho']),
    'u':   add_margin(df['u']),
    'p':   add_margin(df['p']),
    'T':   add_margin(df['T'])
}

# 4. Inizializzazione Curve (t=0)
t0 = times[0]
d0 = df[df['time'] == t0]

l_rho, = ax_rho.plot(d0['x'], d0['rho'], 'r-', lw=2)
ax_rho.set_title(r'Densità ($\rho$)')
ax_rho.set_ylabel('kg/m$^3$')
ax_rho.set_ylim(ylims['rho'])
ax_rho.grid(True, alpha=0.3)

l_u, = ax_u.plot(d0['x'], d0['u'], 'b-', lw=2)
ax_u.set_title('Velocità (u)')
ax_u.set_ylabel('m/s')
ax_u.set_ylim(ylims['u'])
ax_u.grid(True, alpha=0.3)

l_p, = ax_p.plot(d0['x'], d0['p'], 'g-', lw=2)
ax_p.set_title('Pressione (p)')
ax_p.set_ylabel('Pa')
ax_p.set_ylim(ylims['p'])
ax_p.grid(True, alpha=0.3)

# --- NUOVO GRAFICO TEMPERATURA ---
l_T, = ax_T.plot(d0['x'], d0['T'], 'orange', lw=2)
ax_T.set_title('Temperatura (T)')
ax_T.set_ylabel('Kelvin')
ax_T.set_ylim(ylims['T'])
ax_T.grid(True, alpha=0.3)

# Titolo globale col tempo
fig.suptitle(f'Time: {t0:.5f} s', fontsize=16)

# 5. Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Step', 0, n_steps - 1, valinit=0, valstep=1, color='teal')

# 6. Update
def update(val):
    idx = int(slider.val)
    current_time = times[idx]
    dt = df[df['time'] == current_time]
    
    l_rho.set_ydata(dt['rho'])
    l_u.set_ydata(dt['u'])
    l_p.set_ydata(dt['p'])
    l_T.set_ydata(dt['T']) # Aggiorna Temperatura
    
    fig.suptitle(f'Time: {current_time:.5f} s', fontsize=16)
    fig.canvas.draw_idle()

slider.on_changed(update)

print("Plotting avviato...")
plt.show()
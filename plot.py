import pandas as pd
import matplotlib.pyplot as plt

# Leggi i dati
df = pd.read_csv("results.csv")

# Crea il grafico
plt.figure(figsize=(10, 8))

# 1. Densità
plt.subplot(3, 1, 1)
plt.plot(df['x'], df['rho'], 'r-', linewidth=2, label='Densità')
plt.ylabel(r'$\rho$ (kg/m$^3$)')
plt.title('Tubo d\'Urto (Sod Shock Tube)')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Velocità
plt.subplot(3, 1, 2)
plt.plot(df['x'], df['u'], 'b-', linewidth=2, label='Velocità')
plt.ylabel('u (m/s)')
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Pressione
plt.subplot(3, 1, 3)
plt.plot(df['x'], df['p'], 'g-', linewidth=2, label='Pressione')
plt.ylabel('p (Pa)')
plt.xlabel('Posizione x (m)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
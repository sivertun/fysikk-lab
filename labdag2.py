#Innlesing av eksperimentelle verdier for t, x og y


#Plotting av numerisk og eksperimentell bane

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("txy_1.txt", sep="	")
t_exp = np.array(data["t"])
x_exp = np.array(data["x"])
y_exp = np.array(data["y"])

plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Numerisk bane')
plt.plot(x_exp, y_exp, 'ro', label='Eksperimentell bane', markersize=3)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Sammenligning av numerisk og eksperimentell bane')
plt.legend()
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()



#Beregning av eksperimentell v_x, v_y og v

#Figurer som sammenligner numerisk og eksperimentell v(x), v(t) og x(t)

vx_exp = np.gradient(x_exp, t_exp)
vy_exp = np.gradient(y_exp, t_exp)
v_exp = np.sqrt(vx_exp**2 + vy_exp**2)

plt.figure(figsize=(12, 5))
plt.plot(x, v_total, label='Numerisk $v(x)$')
plt.plot(x_exp, v_exp, 'ro', label='Eksperimentell $v(x)$', markersize=3, alpha=0.6)
plt.xlabel('x (m)')
plt.ylabel('v (m/s)')
plt.title('Sammenligning: Hastighet mot posisjon')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(t, v_total, label='Numerisk $v(t)$')
plt.plot(t_exp, v_exp, 'ro', label='Eksperimentell $v(t)$', markersize=3, alpha=0.6)
plt.xlabel('t (s)')
plt.ylabel('v (m/s)')
plt.title('Sammenligning: Hastighet mot tid')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(t, x, label='Numerisk $x(t)$')
plt.plot(t_exp, x_exp, 'ro', label='Eksperimentell $x(t)$', markersize=3, alpha=0.6)
plt.xlabel('t (s)')
plt.ylabel('x (m)')
plt.title('Sammenligning: Posisjon mot tid')
plt.legend()
plt.grid()
plt.show()



#Beregning av eksperimentell Etrans_exp, Erot_exp, Ekin_exp, Epot_exp, Etot_exp

#Figurer som sammenligner numerisk og eksperimentell kinetisk, potensiell og total mekanisk energi


Etrans_exp = 0.5 * M * v_exp**2
Erot_exp = 0.5 * c * M * v_exp**2
Ekin_exp = Etrans_exp + Erot_exp
Epot_exp = M * g * y_exp
Etot_exp = Ekin_exp + Epot_exp

plt.figure(figsize=(12, 8))

plt.plot(x, E_kin, 'b-', label='Numerisk $E_{kin}$')
plt.plot(x_exp, E_kin_exp, 'b.', label='Eksperimentell $E_{kin}$', alpha=0.5)

plt.plot(x, E_pot, 'g-', label='Numerisk $E_{pot}$')
plt.plot(x_exp, Epot_exp, 'g.', label='Eksperimentell $E_{pot}$', alpha=0.5)

plt.plot(x, E_tot, 'r-', label='Numerisk $E_{tot}$')
plt.plot(x_exp, Etot_exp, 'r.', label='Eksperimentell $E_{tot}$', alpha=0.5)

plt.xlabel('x (m)')
plt.ylabel('Energi (J)')
plt.title('Sammenligning: Numeriske og eksperimentelle energier')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()



sluttfarter = [1.114, 1.108, 1.151, 1.136, 1.143, 1.097, 1.212, 1.142]




#Beregning av Delta E = E_i - E_f for 8 vellykkede rulleforsøk

#Beregning av middelverdi og standardfeil for Delta E

#Utskrift av Delta E på formen Delta E = (Middelverdi +- Standardfeil) mJ

Ei_list = np.array([])
Ef_list = np.array([])

deltaE = (Ei_list - Ef_list) * 1000

middelverdi = np.mean(deltaE)
standardavvik = np.std(deltaE, ddof=1)
standardfeil = standardavvik / np.sqrt(len(deltaE))

print(f"Delta E = ({middelverdi:.1f} +- {standardfeil:.1f}) mJ")
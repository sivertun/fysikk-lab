#Numerisk beregning og plotting av de fysiske størrelsene v(x), N(x), f(x), |f/N|, v(t), Epot(x), Ekin(x), Etot(x)
# Bakketoppens høyeste punkt (m):  0.243                                                     
# Banens laveste punkt (m):  0.165
# Helningsvinkel i startposisjonen (grader): -12.8
# Banens maksimale helningsvinkel (grader): 21.8
# De 8 festepunkthøydene (m): [0.3 0.246 0.175 0.176 0.227 0.239 0.198 0.196]
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

#Tallverdier. Tryggest med SI-enheter fra start til mål!
M = 0.031    #kg
g = 9.81     #m/s**2
c = 2/5
h = 0.200    #m
xfast=np.asarray([0,1,2,3,4,5,6,7])*h
xmin = 0
xmax = 1.401
dx = 0.001
x = np.arange(xmin, xmax, dx)
Nx = len(x)

#Skruehøyder:
yfast = np.zeros(8)
#yfast[0] = 0.300
#yfast[1] = yfast[0] - np.random.randint(40,60)/1000
#yfast[2] = yfast[1] - np.random.randint(70,90)/1000
#yfast[3] = yfast[2] + np.random.randint(-30,10)/1000
#yfast[4] = yfast[3] + np.random.randint(30,70)/1000
#yfast[5] = yfast[4] + np.random.randint(-20,20)/1000
#yfast[6] = yfast[5] - np.random.randint(40,80)/1000
#yfast[7] = yfast[6] + np.random.randint(-40,40)/1000

#Kjør denne cellen så mange ganger dere vil, inntil dere får en baneform dere er fornøyd med.
#Når endelig baneform er valgt:
#Sett inn skruehøydene med 3 desimaler (m) og fjern kommentarsymbolet ("#") i neste linje.
yfast = np.asarray([0.300,0.246,0.175,0.176,0.227,0.239,0.198,0.196])
#Når endelig baneform er valgt, kan dere gjerne legge inn # i starten på de 8 linjene som beregner yfast[] ovenfor

#Beregninger:
# CubicSpline tar inn de åtte festepunktene og interpolerer mellom dem slik at vi får en glatt kurve mellom festepunktene. 
# Den glatte kurven beskriver da posisjonen til kulen. Funksjonen som CubicSpline gir ut (her kalt cs) kan ta i mot et array med
# x-posisjoner som argument. Da vil den gi ut de tilsvarene y-posisjonene langs den glatte banen. 
cs = CubicSpline(xfast, yfast, bc_type='natural')
#y = baneformen y(x)
y = cs(x)
#dydx = dy/dx = y'(x) (dimensjonsløs)
dydx = cs(x,1)
#d2ydx2 = y''(x) (enhet 1/m)
d2ydx2 = cs(x,2)
#K = 1/R = banens krumning (1/m)
K = d2ydx2/(1+dydx**2)**(1.5)
#beta = banens helningsvinkel (rad)
beta = np.arctan(dydx)
#betadeg = banens helningsvinkel (grader)
betadeg = beta*180/np.pi
#startvinkel = helningsvinkel i startposisjonen (grader)
startvinkel = betadeg[0]
#maksvinkel = banens maksimale helningsvinkel, i absoluttverdi (grader)
maksvinkel = np.max(np.abs(betadeg))

y37 = y[400:1400]
y27 = y[200:1400]
y37min = np.min(y37)
y37max = np.max(y37)
y27min = np.min(y27)
y27max = np.max(y27)
print('Bakketoppens høyeste punkt (m): %6.3f' %y37max)
print('Banens laveste punkt (m): %6.3f' %y27min)
print('Helningsvinkel i startposisjonen (grader): %4.1f' %startvinkel)
print('Banens maksimale helningsvinkel (grader): %4.1f' %maksvinkel)
print('De 8 festepunkthøydene (m):', yfast)

# ==========================
# ==========================
# START PÅ EGEN KODE
# ==========================
# ==========================



# 1. Hastighet v(x) (Total fart langs banen)
v_total = np.sqrt((10 * g * (y[0] - y)) / 7)

# 2. Beregning av tid t(n)
vx = v_total * np.cos(beta)

# Beregner gjennomsnittlig vx mellom hvert punkt (1400 intervaller)
vx_avg = 0.5 * (vx[:-1] + vx[1:])

# dt = dx / vx_avg
dt = dx / vx_avg

# Kumulativ sum for å finne tidspunktene t0, t1, ..., t1400
t = np.zeros(Nx)
t[1:] = np.cumsum(dt)

# 3. Normalkraft N(x) og Friksjon f(x)
N = M * g * np.cos(beta) + M * v_total**2 * K

# f = (c / (1+c)) * mg sin(beta) -> For kule: (2/7) * mg sin(beta)
f = (2/7) * M * g * np.sin(beta)

# Forholdet |f/N|
f_over_N = np.abs(f / N)

# 4. Energier
E_pot = M * g * y
E_kin = 0.5 * M * v_total**2 * (1 + c)
E_tot = E_pot + E_kin

# Figur 1: Hastighet mot x
plt.figure(figsize=(12, 4))
plt.plot(x, v_total, label='$v(x)$', color='blue')
plt.title('Kulas hastighet $v(x)$')
plt.xlabel('$x$ (m)')
plt.ylabel('$v$ (m/s)')
plt.grid()
plt.show()

# Figur 2: v(t) og x(t)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(t, v_total, color='red')
ax1.set_title('Hastighet mot tid $v(t)$')
ax1.set_xlabel('$t$ (s)')
ax1.set_ylabel('$v$ (m/s)')
ax1.grid()

ax2.plot(t, x, color='green')
ax2.set_title('Posisjon mot tid $x(t)$')
ax2.set_xlabel('$t$ (s)')
ax2.set_ylabel('$x$ (m)')
ax2.grid()
plt.show()

# Figur 3: Krefter N(x) og f(x)
plt.figure(figsize=(12, 5))
plt.plot(x, N, label='Normalkraft $N$', color='black')
plt.plot(x, f, label='Friksjonskraft $f$', color='orange')
plt.title('Krefter som virker på kula')
plt.xlabel('$x$ (m)')
plt.ylabel('Kraft (N)')
plt.legend()
plt.grid()
plt.show()

# Figur 4: Friksjonsforhold |f/N|
plt.figure(figsize=(12, 4))
plt.plot(x, f_over_N, color='purple')
plt.title('Forholdet $|f/N|$')
plt.xlabel('$x$ (m)')
plt.ylabel('$|f/N|$')
plt.grid()
plt.show()

# Figur 5: Energier
plt.figure(figsize=(12, 6))
plt.plot(x, E_pot, label='Potensiell energi $E_{pot}$')
plt.plot(x, E_kin, label='Total kinetisk energi $E_{kin}$')
plt.plot(x, E_tot, '--', label='Total mekanisk energi $E_{tot}$', color='red')
plt.title('Energibevaring langs banen')
plt.xlabel('$x$ (m)')
plt.ylabel('Energi (J)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()

print(f"Kulas rulletid: {t[-1]:.3f}s")
print(f"Slutthastighet: {v_total[-1]:.3f}m/s")
print(f"Total mekanisk energi: {E_tot[-1]:.3f}J")
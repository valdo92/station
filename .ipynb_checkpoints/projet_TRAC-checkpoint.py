## Importation
import math as ma
import matplotlib.pyplot as plt
import numpy as np


# #Définition des variables

#Variables liées à la partie mécanique
M = 45000  #(+15000) en kg, pour une rame
g = 9.81
v_croisiere = 70/3.6#vitesse de croisière en m/s
acc = 0.8 # accélération en m/s^2
d = 2000 # distance en m entre deux sous stations
#alpha = [0 for i in range(400)] + [ma.atan(0.06) for i in range(100)] + [0 for i in range(400)] + [ma.atan(-0.06) for i in range(100)] + [0 for i in range(500)]
alpha = [0 for i in range(d)]
z=0
profil_terrain=[]
for i in range(len(alpha)) :
    z+=ma.tan(alpha[i])
    profil_terrain.append(z)

#Variables liées à la partie électrique
V0 = 835  # en V
Vcat_ini =0.1  # en V
Rs1 = 33 * 10 ** (-3)  # résistance interne des sous stations en Ohm
Rs2 = Rs1
Rlin = 0.1 * 10 ** (-3)  # résistance linéique cable entre SS1 et train (en Ohm par m)

#Variables liées à la partie informatique
N = 100 # discrétisatipn : correspond au temps que 
#met le train entre 2 sous-stations
tau = 1  # pas de temps (en s)
pas_dist = 1 # pas de distance, en m
epsilon = 1e-6 # précision pour la dichotomie


# # Calcul de la vitesse, position et accélération en fonction du temps
# Hypothèse : profil de vitesse trapézoïdal, vitesse de croisière de 20 m/s. On accélère et on freine à +- 0.8 m/s^2.

n = int(v_croisiere/acc) # nombre de pas où le train accélère/deccélère
#Temps
T = [i*tau for i in range(N)]

#Vitesse (t)
V_t = [0]
X_t = [0]
A_t = [0]
k=0
l=0

while True:
    if V_t[k] < 35/3.6:
        t=70000
        a=(t-(1000 + 2.5 * V_t[len(V_t)-1] + 0.023 * V_t[len(V_t)-1] ** 2))/M
        A_t.append(a)
        v=V_t[len(V_t)-1]+tau*a
        #-g*M*ma.sin(alpha[k])
        V_t.append(v)
        k+=1
    elif 35/3.6 <= V_t[len(V_t)-1] < 70/3.6:
        t=(70000*35/3.6)/V_t[len(V_t)-1]
        a=(t-(1000 + 2.5 * V_t[len(V_t)-1] + 0.023 * V_t[len(V_t)-1] ** 2))/M
        v=V_t[len(V_t)-1]+tau*a
        V_t.append(v)
        A_t.append(a)
        
    elif V_t[len(V_t)-1] >= 70/3.6:
        t=0
        a=(t-(1000 + 2.5 * V_t[len(V_t)-1] + 0.023 * V_t[len(V_t)-1] ** 2))/M
        v=V_t[len(V_t)-1]+tau*a
        V_t.append(v)
        A_t.append(a)
        
    X_t.append(X_t[len(X_t)-1]+V_t[len(V_t)-1]*tau)

    d_freinage=-0.6*(V_t[len(V_t)-1]/1.2)**2 + V_t[len(V_t)-1]*(V_t[len(V_t)-1]/1.2)
    
    if d_freinage> 2000-X_t[len(X_t)-1]:
        while V_t[len(V_t)-1]>0:
            V_t.append(V_t[len(V_t)-1]-1.2*tau)
            A_t.append(a)
            
            if V_t[len(V_t)-1]<0:
                V_t.pop()
                V_t.append(0)
                print(V_t)
                X_t.append(X_t[len(X_t)-1]+V_t[len(V_t)-1]*tau)
                
                break
        break

# Fonction construction profil des vitesses







# plt.figure()
# plt.subplot(1,3,1)
# plt.plot(T, X_t,label="distance")
# plt.subplot(1,3,2)
# plt.plot(T, V_t,label="Vitesse")
# plt.subplot(1,3,3)
# plt.plot(T, a,label="Accélération")
# plt.xlabel("Temps (s)")
# plt.legend()
# plt.show()
# # Passage en distance
# Toutes les listes précédentes sont discrétisées en fonction du temps. Pour la suite de l'étude, nous allons les discrétisées en distance (pas de 1 mètre)

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
N = 100 # discrétisatipn : correspond au temps que met le train entre 2 sous-stations
tau = 1  # pas de temps (en s)
pas_dist = 1 # pas de distance, en m
epsilon = 1e-6 # précision pour la dichotomie


# # Calcul de la vitesse, position et accélération en fonction du temps
# Hypothèse : profil de vitesse trapézoïdal, vitesse de croisière de 20 m/s. On accélère et on freine à +- 0.8 m/s^2.

n = int(v_croisiere/acc) # nombre de pas où le train accélère/deccélère
#Temps
T = [i*tau for i in range(N)]

#Accélération (t)
A_t = []
for i in range(n):
    A_t.append(acc)
for i in range(N-2*n):
    A_t.append(0)
for i in range(n):
    A_t.append(-acc)

#Vitesse (t)
V_t = []
#print(n)
for i in range(n):
    V_t.append(acc * tau * i)
for i in range(N-2*n):
    V_t.append(v_croisiere)
for i in range(n):
    V_t.append(v_croisiere - acc * tau * i)

#Position (t)
X_t = []
position = 0
for i in range(N):
    position = position + V_t[i] * tau
    X_t.append(position)

plt.figure()
plt.subplot(1,3,1)
plt.plot(T, X_t,label="distance")
plt.subplot(1,3,2)
plt.plot(T, V_t,label="Vitesse")
plt.subplot(1,3,3)
plt.plot(T, A_t,label="Accélération")
plt.xlabel("Temps (s)")
plt.legend()
plt.show()
## Passage en distance
#Toutes les listes précédentes sont discrétisées en fonction du temps. Pour la suite de l'étude, nous allons les discrétisées en distance (pas de 1 mètre)
A = []
V = []
X = []

dist_acc = int(v_croisiere*(v_croisiere/acc)/2) # distance d'acc et de freinage = 250m (aire sous la courbe de vitesse) Cela représente le nouveau "n"

for i in range(dist_acc):
    X.append(i*pas_dist)
    A.append(acc)
    V.append(acc*ma.sqrt(2/acc)*ma.sqrt(i))
for i in range(dist_acc,d-dist_acc):
    X.append(i*pas_dist)
    A.append(0)
    V.append(v_croisiere)
for i in range(dist_acc):
    X.append(d-dist_acc+i*pas_dist)
    A.append(-acc)
    V.append(v_croisiere - acc*ma.sqrt(2/acc)*ma.sqrt(i))


##Calcul Puissance Train
def frottements():
    a = 2.5
    b = 0.023
    l = []
    for i in range(len(V)):
        l.append(1000 + a * V[i] + b * V[i] ** 2)
    return l

def Puissance_Train():
    P = []
    for i in range(d-dist_acc):
        P.append((M * A[i] + M * g * ma.sin(alpha[i]) + frottements()[i]) * V[i])
    for i in range(d-dist_acc,d-1): #Calcul Puissance récupérée par freinage
        P.append(-0.20*(0.5*M*(V[i]**2-V[i+1]**2))/tau) #On récupère 20% de l'énergie cinétique, et on prends l'accélération et la vitesse au point juste avant de freiner
    return P

P_train = Puissance_Train()
P_train.append(0)
print(P_train)
#P2=P[:d-dist_acc]
#for i in range(d-dist_acc,d-1):
#    P2.append(0)


# # Résolution numérique pour trouver Vcat, Is1 et Is2

#Dichotomie
a = 0.1 # Attention il y a deux zéros dans la fonction recherchée (pour d1=500m). Ici c'est l'intervalle pour le premier zéro.
b = 1000

def dichotomie(f, a, b, epsilon):
    m = (a + b)/2
    c=0
    while abs(a - b) > epsilon:
        if f(a)*f(m) > 0:
            a = m
        else:
            b = m
        m = (a + b)/2
        c+=1
    if V0- m >= 900:
        m=V0-900
    return m

# On trouve Vcat = 1.4035875878762452 Volt
# Vcat = 1.4036
# Is1 = (V0-Vcat) / (Rlin*d1 + Rs1)
# Is2 = (V0-Vcat) / (Rlin*d2 + Rs2)
# On trouve Is1=7718 A et Is2=7186 A

TensionCat = []
for d1 in range(d):
    def f(Vcat):
        return((V0-Vcat)/(Rlin*d1 + Rs1) + (V0-Vcat)/(Rlin*(d-d1) + Rs2) - P_train[d1]/(V0-Vcat))
    TensionCat.append(dichotomie(f,a,b,epsilon))

plt.figure()
plt.subplot(1,2,1)
plt.plot(X, TensionCat,label="Vcat en fonction de la position du train")
plt.legend()
plt.subplot(1,2,2)
plt.plot(X, profil_terrain,label="Profil du terrain")
plt.ylim([-2,12])
plt.legend()
plt.show()
#print(TensionCat)


##Calcul Puissance Électrique
U_train = []
I_train = []
W_circuit = []

for i in range(d):
    
    d1 = i
    Vcat = TensionCat[i] 
    U= V0-Vcat
    Is1 = (V0-Vcat) / (Rlin*d1 + Rs1)
    Is2 = (V0-Vcat) / (Rlin*(d-d1) + Rs2)
    Is = Is1 + Is2
    if U < 500 and Is < 0:
        Is=0
    if U > 600 and Is < 1000:
        Is = 1000
    if U < 600 and U > 500 and Is < (U-500)*100:
        Is = (U-500)*1000
    U_train.append(V0 - Vcat)
    I_train.append(Is)
    W_circuit.append(U_train[i] * I_train[i])
print(W_circuit)

plt.figure()
plt.plot(X, W_circuit,label="Puissance fournie par le réseau")
plt.legend()
plt.show()

##Calcul des courbes
plt.figure()
plt.plot(X, W_circuit,label="Puissance fournie par le réseau")
plt.xlabel("Distance (m)")
plt.ylabel("Puissances (W)")
plt.legend()
plt.plot(X, P_train,label="Puissance nécessaire pour faire avancer le train")
plt.legend()
plt.show()

# plt.xlabel("Position du train")
# plt.ylabel("Puissance nécessaire pour faire avancer le train")

# #Deux trains


## Importation
import math as ma
import matplotlib.pyplot as plt
import numpy as np


##Définition des variables

#Variables liées à la partie mécanique
M = 140000  # en kg, pour une rame
g = 9.81
v_croisiere = 20 #vitesse de croisière en m/s
acc = 0.8 # accélération en m/s^2
d = 1500 # distance en m entre deux sous stations
alpha = [0 for i in range(400)] + [ma.atan(0.06) for i in range(100)] + [0 for i in range(400)] + [ma.atan(-0.06) for i in range(100)] + [0 for i in range(500)]
#alpha = [0 for i in range(d)]
z=0
profil_terrain=[]
for i in range(len(alpha)) :
    z+=ma.tan(alpha[i])
    profil_terrain.append(z)

#Variables liées à la partie électrique
V0 = 835  # en V
Vcat_ini = 0.1 # en V
Rs1 = 0.1  # résistance interne des sous stations en Ohm
Rs2 = Rs1
Rlin = 0.016 * 10 ** (-3)  # résistance linéique cable entre SS1 et train (en Ohm par m)

#Variables liées à la partie informatique
N = 100 # discrétisatipn : correspond au temps que met le train entre 2 sous-stations
tau = 1  # pas de temps (en s)
pas_dist = 1 # pas de distance, en m
epsilon = 1e-6 # précision pour la dichotomie


## Calcul de la vitesse, position et accélération en fonction du temps
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
plt.xlabel("Temps (s)")
plt.legend()
plt.subplot(1,3,2)
plt.plot(T, V_t,label="Vitesse")
plt.xlabel("Temps (s)")
plt.legend()
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
    a = 0.3
    b = 0.1
    l = []
    for i in range(len(V_t)):
        l.append(1000 + a * V_t[i] + b * V_t[i] ** 2)
    return l

def Puissance_Train():
    P_t = []
    for i in range(N-n):
        P_t.append((M * A_t[i] + M * g * ma.sin(alpha[i]) + frottements()[i]) * V_t[i])
    for i in range(N-n,N-1): #Calcul Puissance récupérée par freinage
        P_t.append(-0.20*(0.5*M*(V_t[i]**2-V_t[i+1]**2))/tau) #On récupère 20% de l'énergie cinétique, et on prends l'accélération et la vitesse au point juste avant de freiner
    return P_t

P_t_train = Puissance_Train()
P_t_train.append(0)
print(P_t_train)
#P2=P[:d-dist_acc]
#for i in range(d-dist_acc,d-1):
#    P2.append(0)


## Résolution numérique pour trouver Vcat, Is1 et Is2

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
    return m

#On trouve Vcat = 1.4035875878762452 Volt
#Vcat = 1.4036
#Is1 = (V0-Vcat) / (Rlin*d1 + Rs1)
#Is2 = (V0-Vcat) / (Rlin*d2 + Rs2)
#On trouve Is1=7718 A et Is2=7186 A

TensionCat_t = []
for t in range(N):
    def f(Vcat):
        return((V0-Vcat)/(Rlin*X_t[t] + Rs1) + (V0-Vcat)/(Rlin*(d-X_t[t]) + Rs2) - P_t_train[t]/(V0-Vcat))
    TensionCat_t.append(dichotomie(f,a,b,epsilon))

plt.figure()
#plt.subplot(1,2,1)
plt.plot(T, TensionCat_t,label="Vcat en fonction du temps")
plt.legend()
#plt.subplot(1,2,2)
#plt.plot(X, profil_terrain,label="Profil du terrain")
#plt.ylim([-2,12])
#plt.legend()
plt.show()
#print(TensionCat)


##Calcul Puissance Électrique
U_train_t = []
I_train_t = []
W_circuit_t = []

for t in range(N):
    d1 = X_t[t]
    Vcat = TensionCat_t[t]
    Is1 = (V0-Vcat) / (Rlin*d1 + Rs1)
    Is2 = (V0-Vcat) / (Rlin*(d-d1) + Rs2)
    U_train_t.append(V0 - Vcat)
    I_train_t.append(Is1 + Is2)
    W_circuit_t.append(U_train_t[t] * I_train_t[t])
print(W_circuit_t)

plt.figure()
plt.plot(T, W_circuit_t,label="Puissance fournie par le réseau en fonction du temps")
plt.legend()
plt.show()

##Calcul des courbes
plt.figure()
plt.plot(T, W_circuit_t,label="Puissance fournie par le réseau en fonction du temps")
plt.xlabel("Temps (s)")
plt.ylabel("Puissances (W)")
plt.legend()
plt.plot(T, P_t_train,label="Puissance nécessaire pour faire avancer le train en fonction du temps")
plt.legend()
plt.show()

#plt.xlabel("Position du train")
#plt.ylabel("Puissance nécessaire pour faire avancer le train")
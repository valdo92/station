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
alpha = [0 for e in range(d)]

#Variables liées à la partie électriqueTens
V0 = 835  # en V
Vcat_ini = 0.1 # en V
Rs1 = 0.1  # résistance interne des sous stations en Ohm
Rs2 = Rs1
Rlin = 0.016 * 10 ** (-3)  # résistance linéique cable entre SS1 et train (en Ohm par m)

#Variables liées à la partie informatique
N = 500 # discrétisation
tau = 1  # pas de temps
pas_dist = 1 # pas de distance, en m
epsilon = 1e-6 # précision pour la dichotomie


## Calcul de la vitesse, position et accélération en fonction du temps
# Hypothèse : profil de vitesse trapézoïdal, vitesse de croisière de 20 m/s. On accélère et on freine à +- 0.8 m/s^2.

n = int(v_croisiere/acc) # nombre de pas où le train accélère/deccélère

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

P = Puissance_Train()
print(P)
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
TensionCat = []
for d1 in range(d-1):
    def f(Vcat):
        return((V0-Vcat)/(Rlin*d1 + Rs1) + (V0-Vcat)/(Rlin*(d-d1) + Rs2) - P[d1]/Vcat)
    TensionCat.append(dichotomie(f,a,b,epsilon))

plt.figure()
plt.plot(X[1:-1], TensionCat[1:],label="Vcat en fonction de la position du train")
plt.legend()
plt.show()
#print(TensionCat)


##Calcul Puissance Électrique
U_train = []
I_train = []
W_train = []

for i in range(d-1):
    d1 = i
    Vcat = TensionCat[i]
    Is1 = (V0-Vcat) / (Rlin*d1 + Rs1)
    Is2 = (V0-Vcat) / (Rlin*(d-d1) + Rs2)
    U_train.append(V0 - Is2 * (Rs2 + Rlin *(X[-1]- X[i])))
    I_train.append(Is1 + Is2)
    W_train.append(U_train[i] * I_train[i])
print(W_train)

#plt.figure()
#plt.plot(X[:-1], W_train,label="Puissance fournie par le réseau")
#plt.legend()
#plt.show()

##Calcul des courbes
plt.figure()
plt.subplot(1,2,1)
plt.plot(X[:-1], W_train,label="Puissance fournie par le réseau")
plt.legend()
plt.subplot(1,2,2)
plt.plot(X[:-1], P,label="Puissance nécessaire pour faire avancer le train")
plt.legend()
plt.show()

#plt.xlabel("Position du train")
#plt.ylabel("Puissance nécessaire pour faire avancer le train")
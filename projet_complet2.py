## Importation
import math as ma
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca


# #Définition des variables

#Variables liées à la partie mécanique
M = 45000  #(+15000) en kg, pour une rame
g = 9.81
v_croisiere = 70/3.6#vitesse de croisière en m/s
acc = 0.8 # accélération en m/s^2
d = 2000 # distance en m entre deux sous stations
D=400 # distance entre 2 stations
#alpha = [0 for i in range(400)] + [ma.atan(0.06) for i in range(100)] + [0 for i in range(400)] + [ma.atan(-0.06) for i in range(100)] + [0 for i in range(500)]
alpha = [0 for i in range(d)]
z=0
n_station = 20
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

# +
#Vitesse (t)
A_t= [0]
V_t = [0]
X_t = [0]
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
    
    if d_freinage> D-X_t[len(X_t)-1]:
        while V_t[len(V_t)-1]>0:
            V_t.append(V_t[len(V_t)-1]-1.2*tau)
            X_t.append(X_t[len(X_t)-1]+V_t[len(V_t)-1]*tau)
            A_t.append(-1.2)
            
            if V_t[len(V_t)-1]<0:
                V_t.pop()
                V_t.append(0)
                print(V_t)
                X_t.pop()
          
                X_t.append(X_t[len(X_t)-1]+V_t[len(V_t)-1]*tau)
                
                break
        break
# -

N= len(V_t)
print(N)
T = np.arange(0,N,1)
print(len(X_t))
t_ = np.arange(0, N*n_station,1)

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

# +

for k in range(n_station-1):
    
    for i in range (N):
            X_t.append(400*(k+1)+X_t[i])
            V_t.append(V_t[i])
            A_t.append(A_t[i])


plt.plot(X_t,t_)


# -

##Calcul Puissance Train
def frottements():
    a = 2.5
    b = 0.023
    l = []
    for i in range(len(V_t)):
        l.append(1000 + a * V_t[i] + b * V_t[i] ** 2)
    return l

# +


def Puissance_Train():
    P = []
    for i in range(37):
        if X_t[i]< d_freinage:
            P.append((M * A_t[i] + M * g * ma.sin(alpha[i]) + frottements()[i]) * V_t[i])
        else :
   
            P.append(-0.20*(0.5*M*(V_t[i]**2-V_t[i+1]**2))/tau) #On récupère 20% de l'énergie cinétique, et on prends l'accélération et la vitesse au point juste avant de freiner
    
    return P*n_station

# +
P_train = Puissance_Train()

print(len(P_train))
#P2=P[:d-dist_acc]
#for i in range(d-dist_acc,d-1):
#    P2.append(0)
# -


# # Résolution numérique pour trouver Vcat, Is1 et Is2

#minimum fonction
opti = ca.Opti()
Vcat = opti.variable(1)




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
for i in range(len(X_t)):
    d=X_t[i]
    d1=d%D #distance à la dernière sous sation
    def f(Vcat):
        return(((V0-Vcat)/(Rlin*d1 + Rs1) + (V0-Vcat)/(Rlin*(D-d1) + Rs2) - P_train[i]/(V0-Vcat))*((V0-Vcat)/(Rlin*d1 + Rs1) + (V0-Vcat)/(Rlin*(D-d1) + Rs2) - P_train[i]/(V0-Vcat)))
   
    opti.minimize(f(Vcat))
    opti.solver('ipopt')


    sol = opti.solve()
    TensionCat.append(sol.value(Vcat))

print (len(TensionCat))

plt.figure()
plt.subplot(1,2,1)
plt.plot(X_t, TensionCat,label="Vcat en fonction de la position du train")
plt.legend()
# plt.subplot(1,2,2)
# plt.plot(X_t, profil_terrain,label="Profil du terrain")
# plt.ylim([-2,12])
# plt.legend()
plt.subplot(1,2,2)
plt.plot(X_t[-37:], TensionCat[-37:])
plt.show()
#print(TensionCat)


# +
##Calcul Puissance Électrique
U_train = []
I_train = []
W_circuit = []
    
for i in range(len(TensionCat)):
    d=X_t[i]

    d1 = d%D
    Vcat = TensionCat[i] 

    U= Vcat
    Is1 = (V0-Vcat) / (Rlin*d1 + Rs1)
    Is2 = (V0-Vcat) / (Rlin*(D-d1) + Rs2)
    Is = Is1 + Is2
    P=U*Is
#     if U < 500 :
#         Is=0
#     if U > 600 :
#         Is = 1000

#     if U < 600 and U > 500 :
#         Is = (U-500)*100

    U_train.append(U)
    I_train.append(Is) 
    W_circuit.append(U_train[i] * I_train[i])
print(len(W_circuit), len(U_train), len(I_train))

# +
plt.figure()
plt.subplot(2,1,2)
plt.plot(X_t, U_train,label="Vcat")
plt.legend()
plt.subplot(2,1,1)
plt.plot(X_t, I_train,label="Icat ")
plt.legend()
plt.show()

plt.plot(X_t,W_circuit, label = "puissance électrique")
plt.legend()
# -

##Calcul des courbes
plt.figure()
plt.plot(X_t, W_circuit,label="Puissance fournie par le réseau")
plt.xlabel("Distance (m)")
plt.ylabel("Puissances (W)")
plt.legend()
plt.plot(X_t, P_train,label="Puissance nécessaire pour faire avancer le train")
plt.legend()
plt.show()

# plt.xlabel("Position du train")
# plt.ylabel("Puissance nécessaire pour faire avancer le train")

# #Deux trains


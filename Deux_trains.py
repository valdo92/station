## Importation
import math as ma
import matplotlib.pyplot as plt
import numpy as np

## Modélisation du circuit
d_tot = 12000
Circuit_aller = [i for i in range(int(d_tot/2))]
nb_ss = 5

## Pour le train 1
T_1 = [i*tau for i in range(N*(nb_ss-1))]
V_1 = []
A_1 = []
P_train_1 = []
W_circuit_tot =[]
for e in range(nb_ss-1):
    V_1 += V_t
    A_1 += A_t
    P_train_1 += P_t_train
    W_circuit_tot += W_circuit_t
X_1 = []
position = 0
for i in range(N*(nb_ss-1)):
    position = position + V_1[i] * tau
    X_1.append(position)

##Puissance récupérée
P_recup_1 = []
for e in P_train_1:
    if e < 0 :
        P_recup_1.append(abs(e))
    elif e >= 0 :
        P_recup_1.append(0)


## Pour le train 2
# en première hypothèse, on décale juste notre train pour qu'il accélère au moment ou l'autre freine
T_2 = [i*tau for i in range(N*(nb_ss-1))]
V_2 = []
A_2 = []
P_train_2 = []

for e in range(nb_ss-1):
    V_2 += V_t
    A_2 += A_t
    P_train_2 += P_t_train

X_2 = []
position = int(d_tot/2)
for i in range(N*(nb_ss-1)):
    position = position + V_2[i] * tau
    X_2.append(position)

##Puissance récupérée
P_recup_2 = []
for e in P_train_2:
    if e < 0 :
        P_recup_2.append(abs(e))
    elif e >= 0 :
        P_recup_2.append(0)

##Puissance "Dispo"
P_dispo = []
for i in range(len(P_train_1)) :
    P_dispo.append(W_circuit_tot[i]-(P_train_1[i]+P_train_2[i]))

##Calcul des courbes
plt.figure()
plt.subplot(1,3,1)
plt.plot(T_1, P_train_1,label="Puissance nécessaire pour faire avancer le train 1")
plt.xlabel("Temps (s)")
plt.ylabel("Puissances (W)")
plt.legend()
plt.plot(T_1, W_circuit_tot,label="Puissance fournie par le réseau")
plt.legend()
plt.plot(T_1, P_recup_1,label="Puissance récupérée par freinage du train 1")
plt.legend()
plt.subplot(1,3,2)
plt.plot(T_2, P_train_2,label="Puissance nécessaire pour faire avancer le train 2")
plt.xlabel("Temps (s)")
plt.ylabel("Puissances (W)")
plt.legend()
plt.plot(T_2, W_circuit_tot,label="Puissance fournie par le réseau")
plt.legend()
plt.plot(T_2, P_recup_2,label="Puissance récupérée par freinage du train 2")
plt.legend()
plt.subplot(1,3,3)
plt.plot(T_2, P_dispo,label="Puissance disponible")
plt.xlabel("Temps (s)")
plt.ylabel("Puissances (W)")
plt.legend()
plt.show()




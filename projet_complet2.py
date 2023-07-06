## Importation
import math as ma
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import scipy as sc


# # Un Train

# ## Définition des variables

#Variables liées à la partie mécanique
M = 45000  #(+15000) en kg, pour une rame
g = 9.81
v_croisiere = 70/3.6#vitesse de croisière en m/s
acc = 0.8 # accélération en m/s^2
d = 2000 # distance en m entre deux sous stations
D=500 # distance entre 2 stations
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


# ## Calcul de la vitesse, position et accélération en fonction du temps
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
            X_t.append(D*(k+1)+X_t[i])
            V_t.append(V_t[i])
            A_t.append(A_t[i])


plt.plot(t_,X_t)


# -

# ## Calcul des puissances

##Calcul Puissance Train
def frottements():
    a = 2.5
    b = 0.023
    l = []
    for i in range(len(V_t)):
        l.append(1000 + a * V_t[i] + b * V_t[i] ** 2)
    return l


def Puissance_Train():
    P = []
    for i in range(N):
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


plt.plot(X_t, P_train)

# ## Résolution numérique pour trouver Vcat, Is1 et Is2

TensionCat = []
for i in range(len(X_t)):
    d=X_t[i]
    d1=d%D #distance à la dernière sous sation
    def f(Vcat):
        return(((V0-Vcat)/(Rlin*d1 + Rs1) + (V0-Vcat)/(Rlin*(D-d1) + Rs2) - P_train[i]/(Vcat))**2)
       
    res= sc.optimize.minimize(f,800)
    TensionCat.append(res.x)


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
plt.plot(X_t[-N:], TensionCat[-N:])
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
plt.plot(X_t[:400], I_train[:400],label="Icat ")
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

# # Deux trains


def opti_ener(t_attente):
    TensionCat2 = []
    TensionCat1 = []
    I1,I2 = [],[]
    t2 = np.arange(0, N*n_station+t_attente,1)

    l=[0]*t_attente
    P_train1= P_train + l
    P_train2= l + P_train

    X_t_1 = X_t + [X_t[-1]]*t_attente
    X_t_2 = l +X_t


    for i in range(len(X_t_1)):
        d1=X_t_1[i]%D
        d2=X_t_2[i]%D #distance à la dernière sous sation
        delta = abs(X_t_2[i]-X_t_1[i])
        
        if (delta)== 0:
            TensionCat1.append(835)
            TensionCat2.append(835)
            I1.append(0)
            I2.append(0)
        
        elif (delta)<= D:
             
            def f_(x):
                Vcat1 =x[0]
                Vcat2 = x[1]
                return(((V0-Vcat1)/(Rlin*(D-d1) + Rs2) + (V0-Vcat2)/(Rlin*(d2) + Rs1) - P_train1[i]/(Vcat1)-P_train2[i]/Vcat2)**2)
       
            res= sc.optimize.minimize(f_,np.array([800,800]),constraints = ({'type':"ineq",'fun':lambda x: 900 - x[0]},{'type':"ineq",'fun':lambda x: 900 - x[1]}))
            x1,x2 = float(res.x[0]), float(res.x[1])
            TensionCat1.append(x1)
            TensionCat2.append(x2)
            Vcat1,Vcat2 = x1,x2
            Is1 = (V0-Vcat1) / (Rlin*d1 + Rs1)
            Is2 = (V0-Vcat2) / (Rlin*(D-d2) + Rs2)
            I1.append(Is1 + (Vcat2-Vcat1)/(Rlin*delta))
            I2.append(Is1 + Is2 - (Is1 + (Vcat2-Vcat1)/(Rlin*delta)))
            
        elif (delta)> D:
             
            def f_1(Vcat1):
                return(((V0-Vcat1)/(Rlin*d1 + Rs1) + (V0-Vcat1)/(Rlin*(D-d1) + Rs2) - P_train1[i]/(Vcat1))**2)
            def f_2(Vcat2):
                return(((V0-Vcat2)/(Rlin*d1 + Rs1) + (V0-Vcat2)/(Rlin*(D-d1) + Rs2) - P_train2[i]/(Vcat2))**2)
       
            res_1 = (sc.optimize.minimize(f_1,800))
            TensionCat1.append(float(res_1.x))
            res_2 = (sc.optimize.minimize(f_2,800))
            TensionCat2.append(float(res_2.x))
            Vcat1,Vcat2 = float(res_1.x),float(res_2.x)
            
            Is1_1 = (V0-Vcat1) / (Rlin*d1 + Rs1)
            Is2_1 = (V0-Vcat1) / (Rlin*(D-d1) + Rs2)
            Is1_2 = (V0-Vcat2) / (Rlin*d2 + Rs1)
            Is2_2 = (V0-Vcat2) / (Rlin*(D-d2) + Rs2)
            
            I1.append(Is1_1 + Is2_1)
            I2.append(Is1_2 + Is2_2)
    
    #Puissance
    P_elec1 = np.array(I1)*np.array(TensionCat1)
    P_elec2 = np.array(I2)*np.array(TensionCat2)
    P_tot = P_elec1 + P_elec2
    P_tot[(P_tot<0)]=0
    print(int(100-t_attente))
    
    #Affichage Puissance
    plt.figure()
    plt.subplot(3,1,2)
    plt.plot(t2, P_elec1,label="P_elec_1")
    plt.legend()
    plt.subplot(3,1,1)
    plt.plot(t2, P_elec2,label="P_elec_2")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t2, P_tot,label="P_tot")
    plt.legend()
    plt.show()
    
    #Affichage intensité        
    plt.figure()
    plt.subplot(2,1,2)
    plt.plot(t2, I1,label="I1")
    plt.legend()
    plt.subplot(2,1,1)
    plt.plot(t2, I2,label="I2")
    plt.legend()
    plt.show()
    
    #Affichage Tension 
    plt.figure()
    plt.subplot(2,1,2)
    plt.plot(t2, TensionCat1,label="Vcat1")
    plt.legend()
    plt.subplot(2,1,1)
    plt.plot(t2, TensionCat2,label="Vcat2")
    plt.legend()
    plt.show()
    
    return(np.linalg.norm(P_tot))

opti_ener(64)

#Affichage Puissance
    plt.figure()
    plt.subplot(3,1,2)
    plt.plot(t2, P_elec1,label="P_elec_1")
    plt.legend()
    plt.subplot(3,1,1)
    plt.plot(t2, P_elec2,label="P_elec_2")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t2, P_tot,label="P_tot")
    plt.legend()
    plt.show()
    
    #Affichage intensité        
    plt.figure()
    plt.subplot(2,1,2)
    plt.plot(t2, I1,label="I1")
    plt.legend()
    plt.subplot(2,1,1)
    plt.plot(t2, I2,label="I2")
    plt.legend()
    plt.show()
    
    #Affichage Tension 
    plt.figure()
    plt.subplot(2,1,2)
    plt.plot(t2, TensionCat1,label="Vcat1")
    plt.legend()
    plt.subplot(2,1,1)
    plt.plot(t2, TensionCat2,label="Vcat2")
    plt.legend()
    plt.show()


def opti_ener2(t_attente):
        l=[0]*t_attente
        P_train1= P_train + l
        P_train2= l + P_train
        M=np.array(P_train1)+np.array(P_train2)
        M[(M<0)]=0
        return (np.linalg.norm(M))


y=[]
for t in t_[:500]:
    y.append(opti_ener(t))
plt.plot(t_[:500], y)

print (y.index(min (y[1:])))

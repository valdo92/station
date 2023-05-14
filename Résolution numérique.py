##Importation
import numpy as np
import matplotlib.pyplot as plt

##Def des variables
V0 = 835  # en V
Vcat_ini = 0.1 # en V
epsilon = 1e-6 # précision
d = 1500 # distance en m entre deux sous stations
d1 = 500 # distance en m du train à la première sous station
d2 = d - d1 # distance en km du train à la première sous station
Rs1 = 0.1  # résistance interne des sous stations en Ohm
Rs2 = Rs1
Rlin = 0.016 * 10 ** (-3)  # résistance linéique cable entre SS1 et train (en Ohm par m)
PuissanceTrain = 20920 # en W (valeur calculée  la dernière fois, c'est la puissance nécessaire au train pour rouler en vitesse de croisière, sans accélérer)


##Tracé de la fonction dont on veut trouver le zéro
def f(Vcat):
    return((V0-Vcat)/(Rlin*d1 + Rs1) + (V0-Vcat)/(Rlin*(d-d1) + Rs2) - PuissanceTrain/Vcat)

x = np.linspace(Vcat_ini,5000,num=50000)
F = [f(e) for e in x]
plt.figure()
plt.plot(x,F)

plt.xlabel("Vcat")
plt.ylabel("F")
plt.title("Fonction dont on cherche le zéro")
plt.show()



##Dichotomie
a = 2 #il y a deux zéros dans la fonctions recherchée, le premier est bizarre (F tends vers -inf en 0 à cause de Itrain=Puissance/Vcat) je prends donc l'intervalle du deuxième zéro. Sinon j'aurais pris Vcat_ini
b = 1000000

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

#On trouve Vcat = 833.5964119935852 Volt
Vcat = 833.5964
Is1 = (V0-Vcat) / (Rlin*d1 + Rs1)
Is2 = (V0-Vcat) / (Rlin*d2 + Rs2)
print(Is1,Is2)




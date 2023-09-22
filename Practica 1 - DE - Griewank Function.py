# -*- coding: utf-8 -*-

#Hecho por Alexandro Gutierrez Serna
#Si se hacen pruebas a mano con Griewank hay que asegurarse de que los calculos para el coseno se hagan en radianes

import numpy as np
import matplotlib.pyplot as plt

limite_inferior = -100 #-600 MAX en griewank
limite_superior = 100 #600 MAX en griewank

poblacion = 10 #tambien llamadas soluciones del espacio de busqueda
dimensiones = 2

rango = limite_superior - limite_inferior

F =  0.4 #Factor de mutación
cr = 0.2 #Probabilidad de cruzamiento
max_iters = 500 #Numero maximo de iteraciones

fitness = np.zeros([10,])  #Para guardar fitness
u = np.zeros([dimensiones,]) #Para calcular vector cruzado

#Inicializacion
x=rango*np.random.rand(poblacion,dimensiones)+limite_inferior

#Función objetivo
def griewank(X):
    d = X.size
    sum1 = 0
    prod1 = 1

    #Calculamos los valores de la sumatoria y de la multiplicatoria
    for i in range(d):
        xi = X[i]
        sum1 = sum1 + xi **2
        prod1 = prod1 * np.cos(xi/np.sqrt(i+1))

    #conseguimos el valor de la sumatoria, al que se le resta el resultado de la multiplicatoria y al final se le suma 1
    y = sum1/4000 - prod1 + 1
    return y

'''
###Cambiar función a su funcion correspondiente
def ackley(X):
    d = X.size
    c= 2*np.pi
    b = 0.2
    a = 20

    sum1 = 0
    sum2 = 0

    for i in range(d):
        xi = X[i]
        sum1 = sum1 + xi **2
        sum2 = sum2 + np.cos(c*xi)

    term1 = -a * np.exp(-b * np.sqrt(sum1/d))
    term2 = -np.exp(sum2 / d)

    y = term1 + term2 + a + np.exp(1)
    return y
'''

#Evaluación inicial
for solucion in range(poblacion):
    fitness[solucion] = griewank(x[solucion,:])

#Para guardar el mejor fitness de cada iteracion
best_fitness = np.zeros([max_iters,])
#Para graficar la posicion de la mejor solucion
best_fitness_position = np.zeros([dimensiones,])
contador = 0

#"""Proceso iterativo"""
#Itera 500 veces y trata de acercar los valores a 0
                #500
for i in range(max_iters):

                    #10
    for p in range(poblacion):
        #Mutacion
        indices = np.random.permutation(poblacion)

        xr = np.zeros([3,dimensiones])

        for i in range (3):
            xr[i,:] = x[indices[i],:]

        v = xr[0,:]+F*(xr[1,:]-xr[2,:])


        #Cruzamiento
        for j in range (dimensiones):
            j0 = np.random.randint(2)
            r = np.random.rand()

            if j0 == j or r<=cr:
                u[j] = v[j]
            else:
                u[j] = x[p,j]

        fu = griewank (u)

        #Seleccion - Entra al if si el fitness de la nueva solucion es menor al fitness de la solucion anterior
        if fu < fitness[p]:
            x[p,:] = u
            fitness[p] = fu
            #u es la posicion del nuevo fitness agregado
            best_fitness_position = u
            
    #Tomar el minimo de los 10 fitness y guardarlo en un arreglo
    best_fitness[contador] = np.min(fitness)
    contador = contador + 1 

print("Posicion del mejor fitness: ", best_fitness_position)
print("Mejor fitness: ", best_fitness[-1])
#print("lista best", best_fitness)

#"""Tarea: graficar el mejor fitness de cada iteracion"""
plt.plot(best_fitness)
                       #posicion de la mejor solucion
plt.scatter(max_iters, best_fitness[-1], c='red', marker='o', label='El mejor fitness: '+str(best_fitness[-1]))
plt.annotate(f'Posicion: {best_fitness_position}', (max_iters, best_fitness[-1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=5, color='red')

plt.xlabel('Fitnesses')
plt.ylabel('Valores')
plt.title('Gráfica de convergencia')
plt.legend()
plt.show()

#-*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:21:54 2024

@author: mauri
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#Parámetros iniciales
#population_size = 100  #Tamaño de la población
#generations = 50       #Número de generaciones a simular
#init_res_proportion = 0.1  #Proporción inicial de plantas resistentes

#Datos experimentales
inf_RGA = [19.7, 14.6, 0, 60.8, 20.6] #Porcentaje de infeccción en RGA2 lines
inf_noRGA = [68.5, 67.2, 100.0, 87.5] #Porcentaje de infección en el control
mean_RGA = np.mean(inf_RGA) / 100 #Promedio de las diferentes lineas
mean_noRGA = np.mean(inf_noRGA) / 100 #Promedio de control

#Ajustes iniciales
init_res_proportion = 0.01#1 - mean_noRGA
population_size = np.int64(1235e2)
generations = 40

#Costo metabólico de activar los genes
cost = 0.1

fitness_con_hongo=[1-mean_noRGA, 1-mean_RGA - cost] #fitness con hongo presente (resistencia activa)
fitness_sin_hongo=[0.9,0.8] #fitness sin hongo (sin resistencia activa)

#Matriz de pagos con control genético
#Fila: Estado del hongo (1=Presente, 0=Ausente)
#Columna: Tipo de planta (0=No resistente, 1=Resistente)
payoff_matrix = np.array([fitness_con_hongo,fitness_sin_hongo])

def moran_iter(population, payoff_matrix, fungus_present, cost):
    
    fitness_resistant = payoff_matrix[fungus_present, 1]
    fitness_non_resistant = payoff_matrix[fungus_present, 0]
    
    if fungus_present:#Ajustar fitness para reflejar el costo de activación si el hongo está presente
        fitness_resistant -= cost
    resistant_count = np.sum(population)
    non_resistant_count = len(population) - resistant_count

    #Calcular probabilidad de reproducción proporcional al fitness
    total_fitness = (resistant_count * fitness_resistant+non_resistant_count * fitness_non_resistant)
    prob_reproduce_resistant = (resistant_count * fitness_resistant) / total_fitness
    prob_reproduce_non_resistant = (non_resistant_count * fitness_non_resistant) / total_fitness

    #Seleccionar planta
    reproduce = np.random.choice([0, 1], p=[prob_reproduce_non_resistant, prob_reproduce_resistant])
    replace_idx = np.random.randint(0, len(population)) #Seleccionar planta para ser reemplazada (uniformemente al azar)
    
    #Reemplazar planta seleccionada
    population[replace_idx] = reproduce
    return population

#Simulación adaptada con control genético (Moran)
resistant_proportion = []

#Inicializa la población
population = np.random.choice([0, 1], size=population_size, p=[1-init_res_proportion, init_res_proportion])

for generation in tqdm(range(generations)):
    for _ in range(population_size):  #Cada paso equivale a una generación completa (N eventos)
        fungus_present = np.random.choice([0, 1], p=[0.3, 0.7])  #70% de probabilidad de hongo
        population = moran_iter(population, payoff_matrix, fungus_present, cost)
    resistant_proportion.append(np.mean(population))

#%%
plt.figure(figsize=(10,8))
plt.plot(range(generations), resistant_proportion)
plt.xlabel('Generaciones')
plt.ylabel('Proporción de plantas resistentes')
plt.title('Evolución de la resistencia en la población')
plt.show()

plt.figure(figsize=(10,8))
years = np.array(range(generations))*0.5
plt.plot(years, resistant_proportion)
plt.xlabel('Años')
plt.ylabel('Proporción de plantas resistentes')
plt.title('Evolución de la resistencia en la población')
plt.legend()
plt.show()

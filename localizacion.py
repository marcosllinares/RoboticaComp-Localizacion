#! /usr/bin/env python3

# Robótica Computacional
# Grado en Ingeniería Informática (Cuarto)
# Práctica 5:
#     Simulación de robots móviles holonómicos y no holonómicos.

#localizacion.py

import sys
from math import *
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
import time
# ******************************************************************************
# Declaración de funciones

def distancia(a,b):
  # Distancia entre dos puntos (admite poses)
  return np.linalg.norm(np.subtract(a[:2],b[:2]))

def angulo_rel(pose,p):
  # Diferencia angular entre una pose y un punto objetivo 'p'
  w = atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]
  while w >  pi: w -= 2*pi
  while w < -pi: w += 2*pi
  return w

def mostrar(objetivos,ideal,trayectoria):
  # Mostrar objetivos y trayectoria:
  plt.figure('Trayectoria')
  plt.clf()
  plt.ion() # modo interactivo: show no bloqueante
  # Fijar los bordes del gráfico
  objT   = np.array(objetivos).T.tolist()
  trayT  = np.array(trayectoria).T.tolist()
  ideT   = np.array(ideal).T.tolist()
  bordes = [min(trayT[0]+objT[0]+ideT[0]),max(trayT[0]+objT[0]+ideT[0]),
            min(trayT[1]+objT[1]+ideT[1]),max(trayT[1]+objT[1]+ideT[1])]
  centro = [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]
  radio  = max(bordes[1]-bordes[0],bordes[3]-bordes[2])*.75
  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)
  # Representar objetivos y trayectoria
  idealT = np.array(ideal).T.tolist()
  plt.plot(idealT[0],idealT[1],'-g')
  plt.plot(trayectoria[0][0],trayectoria[0][1],'or')
  r = radio * .1
  for p in trayectoria:
    plt.plot([p[0],p[0]+r*cos(p[2])],[p[1],p[1]+r*sin(p[2])],'-r')
    #plt.plot(p[0],p[1],'or')
  objT   = np.array(objetivos).T.tolist()
  plt.plot(objT[0],objT[1],'-.o')
  plt.show()

def calcular_error(real, ideal, balizas):
  """
  Calcula el error entre el robot real e ideal usando solo sensores.
  Retorna el error normalizado (promedio de errores de distancia + error angular).
  NO accede directamente a las variables de posición - solo usa sensores.
  """
  # Obtener medidas sensoriales del robot real (lo único que podemos "observar")
  distancias_real = real.senseDistance(balizas)
  angulo_real = real.senseAngle(balizas)
  
  # Obtener medidas sensoriales del robot ideal (simulación desde su pose actual)
  distancias_ideal = ideal.senseDistance(balizas)
  angulo_ideal = ideal.senseAngle(balizas)
  
  # Calcular error de distancias (acumulado y normalizado)
  error_distancias = 0.0
  for i in range(len(balizas)):
    error_distancias += abs(distancias_real[i] - distancias_ideal[i])
  error_distancias /= len(balizas)  # Normalizar por número de balizas
  
  # Calcular error angular (con wrap-around para ángulos)
  error_angular = angulo_real - angulo_ideal
  while error_angular > pi: error_angular -= 2*pi
  while error_angular < -pi: error_angular += 2*pi
  error_angular = abs(error_angular)
  
  # Error total normalizado (distancia + ángulo)
  error_total = error_distancias + error_angular
  
  return error_total

def localizacion(balizas, real, ideal, centro, radio, step=0.1, mostrar=False):
  """
  Buscar la localización más probable del robot, a partir de su sistema
  sensorial, dentro de una región cuadrada de centro "centro" y lado "2*radio".
  
  Parámetros:
    balizas: lista de posiciones de balizas fijas [[x1,y1], [x2,y2], ...]
    real: robot real (solo accesible por sensores)
    ideal: robot ideal que vamos a reposicionar
    centro: [x, y, theta] centro de la región de búsqueda
    radio: radio de búsqueda (región será 2*radio x 2*radio)
    step: tamaño de la rejilla (más fino = más preciso pero más lento)
    mostrar: si True, muestra mapa de calor de la búsqueda
  """
  # Matriz para almacenar los errores en cada punto (para visualización)
  imagen = []
  
  # Mejor punto encontrado (inicializamos con error infinito)
  min_error = float('inf')
  mejor_x = centro[0]
  mejor_y = centro[1]
  mejor_theta = centro[2]
  
  # Definir orientaciones a probar (rejilla en el espacio de orientaciones)
  # Probamos 10 orientaciones distribuidas uniformemente en [-pi, pi]
  num_orientaciones = 10
  orientaciones = np.linspace(-pi, pi, num_orientaciones, endpoint=False)
  
  # Crear rejilla de búsqueda en Y (filas)
  for y in np.arange(centro[1]-radio, centro[1]+radio+step/2, step):
    fila = []
    # Crear rejilla de búsqueda en X (columnas)
    for x in np.arange(centro[0]-radio, centro[0]+radio+step/2, step):
      # Para cada punto (x,y), probar diferentes orientaciones
      mejor_error_celda = float('inf')
      mejor_theta_celda = centro[2]
      
      for theta in orientaciones:
        # Posicionar robot ideal en el punto de prueba
        ideal.set(x, y, theta)
        
        # Calcular error usando SOLO sensores (no accediendo a pose directamente)
        error = calcular_error(real, ideal, balizas)
        
        # Guardar la mejor orientación para esta celda
        if error < mejor_error_celda:
          mejor_error_celda = error
          mejor_theta_celda = theta
      
      # Guardar el mejor error de esta celda para la visualización
      fila.append(mejor_error_celda)
      
      # Actualizar el mínimo global si encontramos algo mejor
      if mejor_error_celda < min_error:
        min_error = mejor_error_celda
        mejor_x = x
        mejor_y = y
        mejor_theta = mejor_theta_celda
    
    imagen.append(fila)
  
  # Reposicionar el robot ideal en la pose más probable encontrada
  ideal.set(mejor_x, mejor_y, mejor_theta)
  
  # Mostrar o no el mapa de calor
  if mostrar:
    plt.figure('Localizacion')
    plt.clf()
    plt.ion() # modo interactivo
    plt.xlim(centro[0]-radio,centro[0]+radio)
    plt.ylim(centro[1]-radio,centro[1]+radio)
    imagen.reverse()
    plt.imshow(imagen,extent=[centro[0]-radio,centro[0]+radio,\
                              centro[1]-radio,centro[1]+radio])
    balT = np.array(balizas).T.tolist();
    plt.plot(balT[0],balT[1],'or',ms=10)
    plt.plot(ideal.x,ideal.y,'D',c='#ff00ff',ms=10,mew=2)
    plt.plot(real.x, real.y, 'D',c='#00ff00',ms=10,mew=2)
    plt.show()

# ******************************************************************************

# Definición del robot:
P_INICIAL = [0.,4.,0.] # Pose inicial (posición y orientacion)
P_INICIAL_IDEAL = [2, 2, 0]  # Pose inicial del ideal
V_LINEAL  = .7         # Velocidad lineal    (m/s)
V_ANGULAR = 140.       # Velocidad angular   (º/s)
FPS       = 10.        # Resolución temporal (fps)
MOSTRAR   = True       # Si se quiere gráficas de localización y trayectorias

HOLONOMICO = 1
GIROPARADO = 0
LONGITUD   = .2

# Definición de trayectorias:
trayectorias = [
    [[1,3]],
    [[0,2],[4,2]],
    [[2,4],[4,0],[0,0]],
    [[2,4],[2,0],[0,2],[4,2]],
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)]
    ]

# Definición de los puntos objetivo:
if len(sys.argv)<2 or int(sys.argv[1])<0 or int(sys.argv[1])>=len(trayectorias):
  sys.exit(f"{sys.argv[0]} <indice entre 0 y {len(trayectorias)-1}>")
objetivos = trayectorias[int(sys.argv[1])]

# Definición de balizas fijas (puntos de referencia para localización)
# balizas = [[0, 0], [0, 4], [4, 0], [4, 4]]  # Comentado
balizas = objetivos  # Ahora apuntan a lo mismo

# Definición de constantes:
EPSILON = .1                # Umbral de distancia para alcanzar objetivo
DEVIATION_THRESHOLD = 0.3   # Umbral de desviación para relocalizar (metros)
V = V_LINEAL/FPS            # Metros por fotograma
W = V_ANGULAR*pi/(180*FPS)  # Radianes por fotograma

ideal = robot()
ideal.set_noise(0,0,0)   # Ruido lineal / radial / de sensado
ideal.set(*P_INICIAL_IDEAL)     # operador 'splat'

real = robot()
real.set_noise(.01,.01,.1)  # Ruido lineal / radial / de sensado
real.set(*P_INICIAL)

random.seed(0)
tray_real = [real.pose()]     # Trayectoria seguida

tiempo  = 0.
espacio = 0.
#random.seed(0)
random.seed(time.time())
tic = time.time()

# Localización inicial: búsqueda global con rejilla gruesa
# Región amplia centrada en la pose inicial del ideal con radio grande
localizacion(balizas, real, ideal, centro=P_INICIAL_IDEAL, radio=3.0, step=0.2, mostrar=MOSTRAR)

tray_ideal = [ideal.pose()]  # Trayectoria percibida

distanciaObjetivos = []
for punto in objetivos:
  while distancia(tray_ideal[-1],punto) > EPSILON and len(tray_ideal) <= 1000:
    pose = ideal.pose()

    # Comprueba que la vel y vel angular no supera el maximo
    w = angulo_rel(pose,punto)
    if w > W:  w =  W
    if w < -W: w = -W
    v = distancia(pose,punto)
    if (v > V): v = V
    if (v < 0): v = 0

    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01:
        v = 0
      ideal.move(w,v)
      real.move(w,v)
    else:
      ideal.move_triciclo(w,v,LONGITUD)
      real.move_triciclo(w,v,LONGITUD)

    tray_real.append(real.pose())

    # Decidir nueva localización ⇒ nuevo ideal
    # Calcular la desviación actual entre real e ideal (métrica escalar)
    desviacion_actual = calcular_error(real, ideal, balizas)
    
    # Si la desviación supera el umbral, relocalizar con búsqueda local fina
    if desviacion_actual > DEVIATION_THRESHOLD:
      # Radio proporcional al error (más error = búsqueda más amplia)
      radio_busqueda = min(1.0, desviacion_actual * 1.5)
      localizacion(balizas, real, ideal, centro=ideal.pose(), 
                   radio=radio_busqueda, step=0.05, mostrar=False)
    
    tray_ideal.append(ideal.pose())

    if MOSTRAR:
      mostrar(objetivos, tray_ideal, tray_real)  # Representación gráfica
      input() # Pausa para ver la gráfica

    espacio += v
    tiempo  += 1
  # Antes de pasar a un nuevo punto apuntamos distancia a este objetivo
  distanciaObjetivos.append(distancia(tray_real[-1], punto))


toc = time.time()

# # ===== MÉTRICAS REQUERIDAS =====
# # 1. Tiempo de ejecución
# tiempo_total = toc - tic

# # 2. Desviación acumulada (suma de diferencias pose a pose entre real e ideal)
# desviacion_total = 0.0
# for i in range(min(len(tray_real), len(tray_ideal))):
#   # Desviación como suma de diferencias en x, y, theta (escalar)
#   diff = np.subtract(tray_real[i], tray_ideal[i])
#   desviacion_total += abs(diff[0]) + abs(diff[1]) + abs(diff[2])

# # 3. Distancia al objetivo final
# distancia_final = distanciaObjetivos[-1] if distanciaObjetivos else 0.0
# suma_distancias = np.sum(distanciaObjetivos) if distanciaObjetivos else 0.0

# Mostrar resultados
if len(tray_ideal) > 1000:
  print ("<!> Trayectoria muy larga ⇒ quizás no alcanzada posición final.")
print(f"Recorrido: {espacio:.3f}m / {tiempo/FPS}s")
print(f"Distancia real al objetivo final: {distanciaObjetivos[-1]:.3f}m")
print(f"Suma de distancias a objetivos: {np.sum(distanciaObjetivos):.3f}m")
print(f"Tiempo real invertido: {toc-tic:.3f}sg")
if MOSTRAR:
  mostrar(objetivos, tray_ideal, tray_real)  # Representación gráfica
  input() # Pausa para ver la gráfica

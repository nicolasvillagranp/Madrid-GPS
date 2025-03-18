import pandas as pd
import grafo
import re
import math
import time
import sys
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
INFTY = sys.float_info.max


calles = {}
set_calles = set()
count = 0


def crear_dicc(x):
    global calles, set_calles, count
    if x["Literal completo del vial tratado"] not in set_calles:
        set_calles.add(x["Literal completo del vial tratado"])
        calles[x["Literal completo del vial tratado"]] = [
            x["Literal completo del vial que cruza"]]
        if x["Literal completo del vial que cruza"] not in set_calles:
            set_calles.add(x["Literal completo del vial que cruza"])
        return True
    else:
        if x["Literal completo del vial tratado"] not in calles:
            calles[x["Literal completo del vial tratado"]] = [
                x["Literal completo del vial que cruza"]]
            if x["Literal completo del vial que cruza"] not in set_calles:
                set_calles.add(x["Literal completo del vial que cruza"])
                return True
            else:
                if x["Literal completo del vial que cruza"] not in calles:
                    return True
                else:
                    count += 1
                    return False
        else:
            calles[x["Literal completo del vial tratado"]].append(
                x["Literal completo del vial que cruza"])
            if x["Literal completo del vial que cruza"] not in set_calles:
                set_calles.add(x["Literal completo del vial que cruza"])
                return True
            else:
                if x["Literal completo del vial que cruza"] not in calles:
                    return True
                else:
                    count += 1
                    return False


coords = []


def distances(cordt):
    global coords
    coord = (cordt['Coordenada X (Guia Urbana) cm (cruce)'],
             cordt['Coordenada Y (Guia Urbana) cm (cruce)'])
    x = coord[0]
    y = coord[1]
    found = False
    if coords:
        for c in coords:
            if (x <= (c[0] + 10) and x >= (c[0] - 10)) and (y <= (c[1] + 10) and y >= (c[1] - 10)):
                cordt['Coordenada X (Guia Urbana) cm (cruce)'] = c[0]
                cordt['Coordenada Y (Guia Urbana) cm (cruce)'] = c[1]
                found = True
                break
        if not found:
            coords.append(coord)
    else:
        coords.append(coord)
    return cordt


cruces = pd.read_csv('cruces.csv', sep=';', encoding='latin-1')

# Quitar espacios de antes y de después de los nombres de las columnas
cruces = cruces.applymap(lambda x: x.strip() if isinstance(x, str) else x)


# Obtener un diccionario de clave la calle y valor las calles con las que corta y las coordenadas
interes = cruces.loc[:, ["Literal completo del vial tratado", "Literal completo del vial que cruza",
                         "Coordenada X (Guia Urbana) cm (cruce)", "Coordenada Y (Guia Urbana) cm (cruce)"]]
diccionario_calles = interes.groupby(
    'Literal completo del vial tratado').apply(list).to_dict()

for k, v in diccionario_calles.items():
    diccionario_calles[k] = [v[i][1:] for i in range(len(v))]
reducido = cruces.loc[cruces.apply(lambda x: crear_dicc(x), axis=1)].sort_values(
    by=['Coordenada X (Guia Urbana) cm (cruce)', 'Coordenada Y (Guia Urbana) cm (cruce)'], ascending=True)
reducido.loc[:, ["Coordenada X (Guia Urbana) cm (cruce)", "Coordenada Y (Guia Urbana) cm (cruce)"]] = reducido.loc[:, [
    "Coordenada X (Guia Urbana) cm (cruce)", "Coordenada Y (Guia Urbana) cm (cruce)"]].apply(lambda x: distances(x), axis=1)


def unify(x):
    unique = []
    for i in x:
        for c in i:
            if c not in unique:
                unique.append(c)
    return unique


df_v = reducido.groupby(['Coordenada X (Guia Urbana) cm (cruce)', 'Coordenada Y (Guia Urbana) cm (cruce)'])[
    ['Literal completo del vial tratado', 'Literal completo del vial que cruza']].apply(lambda x: x.values.tolist())
df_v = df_v.apply(lambda x: unify(x))


def calcular_dist(u, v):
    return math.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2)


def calcular_min(i, aux):
    calle_min = ''
    posibles = calles[aux]
    min = INFTY
    for j in posibles:
        dist = calcular_dist(posibles[j], i)
        if dist < min and dist != 0:
            min = dist
            calle_min = j
    return calle_min, min


INFTY = sys.float_info.max

cruces = pd.read_csv("cruces.csv", sep=";", encoding='latin-1')
cruces = cruces.applymap(lambda x: x.strip() if isinstance(x, str) else x)
interes = cruces.loc[:, ["Literal completo del vial tratado", "Literal completo del vial que cruza",
                         "Coordenada X (Guia Urbana) cm (cruce)", "Coordenada Y (Guia Urbana) cm (cruce)"]]


calles = {i: {} for i in interes['Literal completo del vial tratado'].unique()}


def crear_calles(x):
    global calles
    coor_x = x['Coordenada X (Guia Urbana) cm (cruce)']
    coor_y = x['Coordenada Y (Guia Urbana) cm (cruce)']
    if x['Literal completo del vial que cruza'] not in calles[x['Literal completo del vial tratado']]:
        calles[x['Literal completo del vial tratado']
               ][x['Literal completo del vial que cruza']] = (coor_x, coor_y)
    if x['Literal completo del vial que cruza'] in calles:
        if x['Literal completo del vial tratado'] not in calles[x['Literal completo del vial que cruza']]:
            calles[x['Literal completo del vial que cruza']
                   ][x['Literal completo del vial tratado']] = (coor_x, coor_y)
    else:
        calles[x['Literal completo del vial que cruza']] = {}
        calles[x['Literal completo del vial que cruza']
               ][x['Literal completo del vial tratado']] = (coor_x, coor_y)


def find_max_speed(nombre_calle):
    tipos = {'AUTOVIA': 100, 'AVENIDA': 90, 'CARRETERA': 70, 'CALLEJON': 30, 'CAMINO': 30, 'ESTACION DE METRO': 20,
             'PASADIZO': 20, 'PLAZUELA': 20, 'COLONIA': 20}
    for i in tipos:
        if re.search(i, nombre_calle):
            return tipos[i]
    return 50


reducido.apply(crear_calles, axis=1)
diccionario_v = df_v.to_dict()


G = grafo.Grafo()
tiempos = grafo.Grafo()
# Creacion de vertices
for i in diccionario_v:
    G.agregar_vertice(i)
    tiempos.agregar_vertice(i)


def find_edge(i, j):
    velocidad = find_max_speed(j)
    u, dist1 = calcular_min(i, j)
    if u != '':
        clave = calles[j].pop(u)
        v, dist2 = calcular_min(i, j)
        calles[j][u] = clave
    # Vemos si el signo es el mismo.
    if u != '' and v != '':
        if np.sign(calles[j][u][0]-i[0]) == np.sign(calles[j][v][0]-i[0]) and np.sign(calles[j][u][1]-i[1]) == np.sign(calles[j][v][1]-i[1]):
            G.agregar_arista(i, calles[j][u], None, dist1)
            tiempos.agregar_arista(i, calles[j][u], None, dist1/velocidad)
        else:
            G.agregar_arista(i, calles[j][u], None, dist1)
            tiempos.agregar_arista(i, calles[j][u], None, dist1/velocidad)
            G.agregar_arista(i, calles[j][v], None, dist2)
            tiempos.agregar_arista(i, calles[j][v], None, dist2/velocidad)
    elif u != '' and v == '':
        G.agregar_arista(i, calles[j][u], None, dist1)
        tiempos.agregar_arista(i, calles[j][u], None, dist1/velocidad)


for i in diccionario_v:
    aux = diccionario_v[i]
    for j in aux:  # Creamos ambos grafos a la vez.
        find_edge(i, j)
direcciones = pd.read_csv("direcciones.csv", sep=";",
                          encoding='latin-1', low_memory=False)
direcciones = direcciones.applymap(
    lambda x: x.strip() if isinstance(x, str) else x)
direcciones['Coordenada X (Guia Urbana) cm'] = direcciones['Coordenada X (Guia Urbana) cm'].str.replace(
    '-', '0')
direcciones['Coordenada Y (Guia Urbana) cm'] = direcciones['Coordenada Y (Guia Urbana) cm'].str.replace(
    '-', '0')
direcciones['Coordenada X (Guia Urbana) cm'] = direcciones['Coordenada X (Guia Urbana) cm'].astype(
    float)
direcciones['Coordenada Y (Guia Urbana) cm'] = direcciones['Coordenada Y (Guia Urbana) cm'].astype(
    float)
direcciones['direccion'] = direcciones['Clase de la via']+' '+direcciones['Partícula de la vía'] + \
    ' '+direcciones['Nombre de la vía'] + ' ' + \
    direcciones['Literal de numeracion']


def get_dir(direcciones):
    calle = input("Ingrese el nombre de la via: ").strip()
    num = input("Ingrese el numero o el km de la via: ").strip()
    for j in direcciones.index:
        i = direcciones.loc[j, 'direccion']
        if re.search(calle, i, re.IGNORECASE):
            if int(re.search('[0-9]+', i).group(0)) == int(num):
                return direcciones.loc[j]


def camino():
    entrada = input(
        'Elige si quieres el camino mas corto o el mas rapido: corto/rapido: ')
    if entrada == 'corto':
        a = G.camino_minimo(coor_or, coor_des)
        Grafo_net = G.convertir_a_NetworkX()
    else:
        a = tiempos.camino_minimo(coor_or, coor_des)
        Grafo_net = tiempos.convertir_a_NetworkX()
    vertices = []
    aux = coor_des
    while aux != coor_or:
        aux = a[aux]
        vertices.append(aux)
    copia = a.copy()
    ######## Instrucciones #########
    vertices_ok = vertices[::-1]
    vertices_ok.pop(0)
    dir = {}
    for i in range(len(vertices_ok)-1):
        a = diccionario_v[vertices_ok[i]]
        b = diccionario_v[vertices_ok[i+1]]
        for j in range(len(a)):
            if a[j] in b:
                dist = round((math.sqrt((vertices_ok[i+1][0]-vertices_ok[i][0])
                                        ** 2+(vertices_ok[i+1][1]-vertices_ok[i][1])**2))/100, 2)
                if a[j] in dir:
                    dir[a[j]] += dist
                else:
                    dir[a[j]] = dist
    for i in dir:
        round(dir[i], 2)
        print(i, 'durante ', dir[i], 'm', 'más adelante gire hacia')
        time.sleep(1)
    print(f'{calle_des} y llegara a su destino')
    ######### GRAFICAR #########
    node_colors = []
    for node in Grafo_net.nodes:
        if node == coor_or or node == coor_des:
            node_colors.append('g')
        elif node in vertices:
            node_colors.append('r')
        else:
            node_colors.append('b')
    posicion = {i: i for i in Grafo_net.nodes}
    nx.draw(Grafo_net, pos=posicion, node_size=5, width=0.1,
            node_color=node_colors)
    plt.show()
    G.eliminar_vertice(coor_or)
    G.eliminar_vertice(coor_des)
    tiempos.eliminar_vertice(coor_or)
    tiempos.eliminar_vertice(coor_des)


if __name__ == "__main__":
    print('Bienvenido al programa de rutas')
    print('Cargando datos...')
    continuar = True
    while continuar:
        print('Se le pediran los datos del origen')
        origen = get_dir(direcciones)
        print('Se le pediran los datos del destino')
        destino = get_dir(direcciones)
        calle_or = cruces[origen['Codigo de via'] == cruces['Codigo de vía tratado']
                          ].iloc[0]['Literal completo del vial tratado']
        coor_or = (origen['Coordenada X (Guia Urbana) cm'],
                   origen['Coordenada Y (Guia Urbana) cm'])
        calle_des = cruces[destino['Codigo de via'] == cruces['Codigo de vía tratado']
                           ].iloc[0]['Literal completo del vial tratado']
        coor_des = (destino['Coordenada X (Guia Urbana) cm'],
                    destino['Coordenada Y (Guia Urbana) cm'])
        G.agregar_vertice(coor_or)
        G.agregar_vertice(coor_des)
        tiempos.agregar_vertice(coor_or)
        tiempos.agregar_vertice(coor_des)
        find_edge(coor_or, calle_or)
        find_edge(coor_des, calle_des)
        camino()
        entrada = input('Desea ingresar otra direccion? (s/n): ')
        if entrada == 'n':
            continuar = False

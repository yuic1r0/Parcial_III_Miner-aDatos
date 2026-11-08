
from networkx import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Librerías cargadas exitosamente.")

# Definición de coordenadas de los puntos q1 a q6. Matriz de distancias cuadradas d^2
puntos = {
    'q1': [1, 3],
    'q2': [3, 4],
    'q3': [4, 1],
    'q4': [7, 2],
    'q5': [8, 6],
    'q6': [6, 8]
}

nombres = list(puntos.keys())
X = np.array(list(puntos.values()))


dist_d2_vec = pdist(X, metric='sqeuclidean')


matriz_completa = pd.DataFrame(squareform(dist_d2_vec), index=nombres, columns=nombres)


matriz_triangular = matriz_completa.astype(object)
for i in range(len(nombres)):
    for j in range(i + 1):
        matriz_triangular.iloc[i, j] = "—"

print("=== MATRIZ INITIAL DE DISTANCIAS CUADRADAS d^2 (PASO 0) ===")
display(matriz_triangular)

#Algoritmo aglomerativo

def simular_agrupamiento_paso_a_paso(X, nombres):
    clusters = [{n} for n in nombres]
    n = len(nombres)
    dist_matrix = squareform(pdist(X, metric='sqeuclidean'))

    curr_labels = [str(c) for c in nombres]
    curr_dists = pd.DataFrame(dist_matrix, index=curr_labels, columns=curr_labels)
    
    print(f"Estado inicial: {len(curr_labels)} clusters singleton.\n")
    
    paso = 1
    while len(curr_labels) > 1:
        min_val = float('inf')
        pair_to_merge = None
        
        for i in range(len(curr_labels)):
            for j in range(i + 1, len(curr_labels)):
                val = curr_dists.iloc[i, j]
                if val < min_val:
                    min_val = val
                    pair_to_merge = (i, j)
        
        i, j = pair_to_merge
        lbl1, lbl2 = curr_labels[i], curr_labels[j]
        new_label = f"{{{lbl1},{lbl2}}}".replace("'", "")
        
        print(f"--- PASO {paso} ---")
        print(f"Fusionando {lbl1} y {lbl2} a una distancia d^2 = {min_val:.0f}")
        
        new_dists = {}
        for idx, lbl in enumerate(curr_labels):
            if idx != i and idx != j:
                d_min = min(curr_dists.iloc[i, idx], curr_dists.iloc[j, idx])
                new_dists[lbl] = d_min
        
        remaining_labels = [lbl for k, lbl in enumerate(curr_labels) if k != i and k != j]
        curr_labels = remaining_labels + [new_label]
        
        m_size = len(curr_labels)
        new_df = pd.DataFrame(np.zeros((m_size, m_size)), index=curr_labels, columns=curr_labels)
        
        for r in remaining_labels:
            for c in remaining_labels:
                new_df.loc[r, c] = curr_dists.loc[r, c]
        
        for r, d_val in new_dists.items():
            new_df.loc[r, new_label] = d_val
            new_df.loc[new_label, r] = d_val
            
        curr_dists = new_df
        paso += 1
        print(f"Clusters activos ({len(curr_labels)}): {curr_labels}\n")

simular_agrupamiento_paso_a_paso(X, nombres)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Librerías cargadas exitosamente.\n")


# Definición de coordenadas de los puntos q1 a q6
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

print("=== MATRIZ INICIAL DE DISTANCIAS CUADRADAS d^2 (PASO 0) ===")
print(matriz_triangular)
print("\n" + "="*70 + "\n")



#Algoritmo Aglomerativo 


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
print("="*70 + "\n")



# Dendrograma 


Z = linkage(dist_d2_vec, method='single')


fig, ax = plt.subplots(figsize=(9, 5))
dend = dendrogram(Z, labels=nombres, color_threshold=0, above_threshold_color='#1a365d', ax=ax)

ax.set_title("Dendrograma de Clustering Jerárquico — Parcial 3 (Fila A)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Usuarios / Puntos", fontsize=11, labelpad=10)
ax.set_ylabel("Distancia Cuadrada ($d^2$)", fontsize=11, labelpad=10)


alturas = [5, 8, 10, 17]
for h in alturas:
    ax.axhline(y=h, color='#c53030', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(x=0.5, y=h + 0.3, s=f"d² = {h}", color='#c53030', fontweight='bold', fontsize=9)

ax.set_ylim(0, 20)
plt.tight_layout()
plt.show()



#Gráfico del Corte del Dendrograma



corte_h = 9.0
labels_4_clusters = fcluster(Z, t=corte_h, criterion='distance')

df_resumen = pd.DataFrame({
    'Punto': nombres,
    'Coordenadas': [str(p) for p in X],
    'Cluster (h=9.0)': labels_4_clusters
})

print("=== ASIGNACIÓN DE CLUSTERS PARA CORTE EN d^2 = 9.0 ===")
print(df_resumen)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X[:, 0], X[:, 1], c=labels_4_clusters, cmap='tab10', s=150, zorder=3)

for i, txt in enumerate(nombres):
    ax.annotate(f" {txt} {X[i]}", (X[i, 0] + 0.15, X[i, 1] + 0.15), fontsize=11, fontweight='bold')

ax.set_title("Plano $\\mathbb{R}^2$ — Asignación con 4 Clusters (Corte a $h=9.0$)", fontsize=12, fontweight='bold')
ax.set_xlabel("$x_1$ (Horas de sesión diaria)", fontsize=10)
ax.set_ylabel("$x_2$ (Géneros distintos por semana)", fontsize=10)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
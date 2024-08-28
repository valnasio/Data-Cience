# Importação das bibliotecas necessárias 
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Configuração do backend do Matplotlib para evitar problemas de renderização
import matplotlib
matplotlib.use('Agg')  # Usa o backend Agg para salvar gráficos como arquivos de imagem

#  Carregar o Conjunto de Dados Wine
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

#  Exploração e Pré-processamento de Dados
df_wine = pd.DataFrame(X_wine, columns=wine.feature_names)
df_wine['target'] = y_wine

# Verificar as características principais
print("\nPrincipais características do conjunto de dados Wine:")
print(df_wine.describe())

# Verificar se há valores ausentes
print("\nVerificação de valores ausentes no conjunto de dados Wine:")
print(df_wine.isnull().sum())

# Normalização dos dados
scaler_wine = StandardScaler()
X_wine_scaled = scaler_wine.fit_transform(X_wine)

#  Aplicação do Algoritmo K-Means (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_wine_scaled)

# Visualização dos Clusters
# Utilizando PCA para reduzir a dimensionalidade e visualizar os clusters em 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_wine_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis')
plt.title("Visualização dos Clusters (K-Means)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")

# Salvar o gráfico como uma imagem em vez de exibi-lo
plt.savefig("clusters_kmeans.png")

# Avaliação dos Clusters
silhouette_avg = silhouette_score(X_wine_scaled, clusters)
print(f"\nSilhouette Score: {silhouette_avg:.2f}")
print(f"Inércia (Soma das Distâncias ao Quadrado dentro dos Clusters): {kmeans.inertia_:.2f}")

# Comparação dos clusters com as classes reais
print("\nComparação dos clusters formados pelo K-Means com as classes reais:")
print(pd.crosstab(y_wine, clusters))

# Discussão dos Resultados
# Análise de como os clusters se comparam com as classes reais e os desafios encontrados

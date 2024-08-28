# Importação das bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# Carregar o Conjunto de Dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Exploração e Pré-processamento de Dados
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Verificar a distribuição das classes
print("Distribuição das classes:")
print(df['target'].value_counts())

# Verificar se há valores ausentes
print("\nVerificação de valores ausentes:")
print(df.isnull().sum())

# Normalização dos dados 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divião dos Dados (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Aplicação do Algoritmo KNN
# Testando difeentes valores de k e selecionando o melhor
ks = [1, 3, 5, 7, 9]
best_k = 1
best_accuracy = 0

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Treinamento final com o melhor k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Avaliação do modelo KNN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

print(f"\nKNN com k={best_k}:\nAcurácia: {best_accuracy:.2f}")
print("Matriz de Confusão (KNN):\n", conf_matrix_knn)
print(f"Precisão: {precision_knn:.2f}, Recall: {recall_knn:.2f}, F1-Score: {f1_knn:.2f}")

# Aplicando  Algoritmo Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Avaliação do modelo
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

print(f"\nNaive Bayes:\nAcurácia: {accuracy_nb:.2f}")
print("Matriz de Confusão (Naive Bayes):\n", conf_matrix_nb)
print(f"Precisão: {precision_nb:.2f}, Recall: {recall_nb:.2f}, F1-Score: {f1_nb:.2f}")

# Desempenho
print("\nComparação de Desempenho entre KNN e Naive Bayes:")
print(f"KNN - Acurácia: {best_accuracy:.2f}, Precisão: {precision_knn:.2f}, Recall: {recall_knn:.2f}, F1-Score: {f1_knn:.2f}")
print(f"Naive Bayes - Acurácia: {accuracy_nb:.2f}, Precisão: {precision_nb:.2f}, Recall: {recall_nb:.2f}, F1-Score: {f1_nb:.2f}")

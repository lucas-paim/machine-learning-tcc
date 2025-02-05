# Inicialização do modelo kNN
knn = KNeighborsClassifier()

# Treinamento do Modelo
knn.fit(X_train, y_train)

# Predição nos Dados de Teste
y_pred = knn.predict(X_test)

# Plotar Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Pain", "Pain"], yticklabels=["No Pain", "Pain"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Avaliação do Modelo
showResults(classification_report(y_test, y_pred), accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred))

# Utilizando o Cross validation
model = KNeighborsClassifier()
scores = cross_val_score(model, X, y, cv=5)
print("Scores de cada fold:", scores)
print("Média dos scores:", scores.mean())

model = KNeighborsClassifier()
scores = cross_val_score(model, X, y, cv=10)
print("Scores de cada fold:", scores)
print("Média dos scores:", scores.mean())

# Pipeline revisado para kNN com dados desbalanceados
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Normalizar
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

# Buscar melhor k
param_grid = {'n_neighbors': list(range(1, 31))}
grid = GridSearchCV(KNeighborsClassifier(weights='distance'), param_grid, scoring='f1', cv=5)
grid.fit(X_train_balanced, y_train_balanced)
knn = grid.best_estimator_

# Previsões
y_pred = knn.predict(X_test)

# Plotar Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Pain", "Pain"], yticklabels=["No Pain", "Pain"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

showResults(classification_report(y_test, y_pred), accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred))

# Utilizando o Cross validation
model = KNeighborsClassifier()
scores = cross_val_score(model, X, y, cv=5)
print("Scores de cada fold:", scores)
print("Média dos scores:", scores.mean())

model = KNeighborsClassifier()
scores = cross_val_score(model, X, y, cv=10)
print("Scores de cada fold:", scores)
print("Média dos scores:", scores.mean())

# Aplicando o Hyperparameter Tuning ========================================================================================================= #

# Encontrando o Melhor Valor de k
error_rates = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    error_rates.append(np.mean(pred_k != y_test.ravel()))

# Plotando os Erros para cada k até 30
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), error_rates, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Aplicando o Melhor Valor de k
best_k = np.argmin(error_rates) + 1
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train.ravel())
y_pred = knn.predict(X_test)

# Plotar Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Pain", "Pain"], yticklabels=["No Pain", "Pain"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Avaliação do Modelo Final
print(f"Best K value: {best_k}")
showResults(classification_report(y_test, y_pred), accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred))

# Aplicando o Parameter Grid ================================================================================================================ #

# Definição do Paramater Grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Número de vizinhos a serem considerados
    'weights': ['uniform', 'distance'],  # Tipo de ponderação dos vizinhos
    'metric': ['euclidean', 'manhattan']  # Métricas de distância a serem testadas
}

# Inicialização do Objeto GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')

# Treinando o Modelo
grid_search.fit(X_train, y_train)

# Mostrando os melhores parametros e o melhor score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Usando o melhor estimador para testar o modelo
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Avaliação do Modelo
showResults(classification_report(y_test, y_pred), accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred))

# Plotar Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Pain", "Pain"], yticklabels=["No Pain", "Pain"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# Inicialização do modelo SVM
model = SVC(kernel='linear', C=1.0, random_state=42, class_weight ='balanced')

# Treinamento do Modelo
model.fit(X_train, y_train)

# Predição nos Dados de Teste
y_pred = model.predict(X_test)

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
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

model = SVC()
scores = cross_val_score(model, X, y, cv=5)  # 5 folds
print("Scores de cada fold:", scores)
print("Média dos scores:", scores.mean())

model = SVC()
scores = cross_val_score(model, X, y, cv=10)  # 10 folds
print("Scores de cada fold:", scores)
print("Média dos scores:", scores.mean())

# Aplicando Grid Search ============================================================================================================== #

# Definição do Paramater Grid
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['linear'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'coef0': [0, 0.5, 1]
}

# Inicialização do Objeto GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')

# Treinando o Modelo
grid_search.fit(X_train, y_train)

# Mostrando os melhores parametros e o melhor score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Usando o melhor estimador para testar o modelo
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Plotar Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Pain", "Pain"], yticklabels=["No Pain", "Pain"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Avaliação do Modelo
showResults(classification_report(y_test, y_pred), accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred))

# Aplicando Feature Selection ======================================================================================================== #

from sklearn.feature_selection import SelectKBest, f_classif

# Selecionando as melhores k features
k = 19
selector = SelectKBest(f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Treinamento do modelo com as melhores features
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train_selected, y_train)

# Predição nos Dados de Teste
y_pred = model.predict(X_test_selected)

# Plotar Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Pain", "Pain"], yticklabels=["No Pain", "Pain"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Avaliação do Modelo
showResults(classification_report(y_test, y_pred, zero_division = 0), accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred))






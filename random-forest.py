# Criar o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo
rf_model.fit(X_train, y_train)

# Fazer previsões
y_pred = rf_model.predict(X_test)

# Gerar a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Pain", "Pain"], yticklabels=["No Pain", "Pain"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Avaliação do Modelo
showResults(classification_report(y_test, y_pred), accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred))

# Utilizando Cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)  # 5 folds
print("Scores de cada fold:", scores)
print("Média dos scores:", scores.mean())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=10)  # 10 folds
print("Scores de cada fold:", scores)
print("Média dos scores:", scores.mean())

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
print(feature_importance_df.sort_values(by="Importance", ascending=False))

scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-Validation Accuracy:", scores.mean())

# Trabalho de Conclusão de Curso
# Lucas Borba Paim
# RA: 117590

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carregando o Dataset
df = pd.read_excel("final_data.xlsx")

features = [
    "band_power_delta", "band_power_theta", "band_power_alpha", "band_power_beta", "band_power_gamma",
    "rbp_delta", "rbp_theta", "rbp_alpha", "rbp_beta", "rbp_gamma",
    "alpha_pf", "alpha_pp", "power_ratio",
    "entropy_delta", "entropy_theta", "entropy_alpha", "entropy_beta", "entropy_gamma", "sef_50"
    ]

X = df[features].values
y = df["pain"].values

# Divisão dos dados de treinamento (70%) e de teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização dos Dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Função para facilitar a exibição dos resultados de cada método
def showResults(classification, accuracy, matrix):
    print("\nClassification Report:")
    print(classification)
    print("Accuracy:", accuracy, "\n")


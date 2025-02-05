from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from keras.layers import Dense
from keras.models import Sequential

# Definir o modelo de rede neural
def define_model(n_input):
    model = Sequential()
    # Primeira camada oculta
    model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    # Camada de saída
    model.add(Dense(1, activation='sigmoid'))
    # Compilar o modelo
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

# Criar o modelo
n_input = trainX.shape[1]
model = define_model(n_input)

# Treinar o modelo
model.fit(trainX, trainy, epochs=100, verbose=0)

# Fazer previsões nos dados de teste
yhat = model.predict(testX)

# Avaliar o ROC AUC
score = roc_auc_score(testy, yhat)
print('ROC AUC: %.3f' % score)

# Utilizando Dropout de 50% ============================================================================ #

from keras.layers import Dropout
# Dividir os dados em treino e teste
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir o modelo de rede neural
def define_model(n_input):
    model = Sequential()
    # Primeira camada oculta
    model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))  # 50% de dropout
    # Camada de saída
    model.add(Dense(1, activation='sigmoid'))
    # Compilar o modelo
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

# Criar o modelo
n_input = trainX.shape[1]
model = define_model(n_input)

# Definir os pesos para lidar com o desbalanceamento das classes
weights = {0: 0.6, 1: 0.39}  # Classe 0 (no pain, minoritária) e classe 1 (pain, majoritária)

# Treinar o modelo
history = model.fit(trainX, trainy, class_weight=weights, epochs=100, verbose=0)

# Fazer previsões nos dados de teste
yhat = model.predict(testX)

# Avaliar o modelo com ROC AUC
score = roc_auc_score(testy, yhat)
print('ROC AUC: %.3f' % score)



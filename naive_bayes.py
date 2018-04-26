import numpy as np
from sklearn.naive_bayes import BernoulliNB

# Dados 4 sintomas e 6 amostras temos a matriz abaixo

# Amostras de teste
# 1 = Sim
# 0 = Não
x = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
])

# Classes
# 1 = Doente
# 0 = Saudável
Y = np.array([1, 1, 1, 0, 0, 0])

# Amostra para predição
n = [[1, 0, 0, 1]]

bnb = BernoulliNB()

# Fato não importante no momento (Nota pessoal: estudar isso depois)
probabilities_test = bnb.fit(x, Y).predict_proba(x)

# Probabilidades normalizadas
# (a soma das mesmas resultam em 1 que seria na prática 100%)
#
# P(Doente|N) = P(Doente|N) / P(Doente|N) + P(Saudável|N)
# P(Saudável|N) = P(Saudável|N) / P(Saudável|N) + P(Doente|N)
normalized_probabilities = bnb.fit(x, Y).predict_proba(n)
print('Normalized probabilities: Class 1={} Class 0={}'.format(
    normalized_probabilities[0, 1],
    normalized_probabilities[0, 0],
))

# Predizendo classe resultado da nossa amostra N
predicted_class = bnb.predict(n)
print('Class: {}'.format(predicted_class))

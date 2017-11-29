from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)  # to make the output consistent, same "randomness" all the time

X = np.array([  # x range 8.5, y range 7.8
    [-5.5, 3.5],
    [0.75, 3.75],
    [0.25, 0.4],
    [3.5, 0.4],
    [-0.4, -0.35],
    [-2.3, -4.3],
    [3, -3]
])
Y = np.array([0, 0, 1, 0, 1, 0, 1])  # blue=0, red=1

# idea: radius as a 3rd input value to enrich input-space
#      -> doesn't seem to help improve the results
#         Bernhard says data needs to be translation invariant
# radiuses = np.array([np.sqrt(x**2 + y**2) for x, y in X])
# X = np.column_stack((X, radiuses))

model = Sequential()
# these three layer-architecture and the activation types were really just experimentally found :)
model.add(Dense(7, input_dim=2, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# no clue what the following parameters are, copied from the keras Doku
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=500)  # why does it have to be so many epochs? no 100% accuracy if you go much lower
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

n = 25
for i in range(0, n + 3):
    for j in range(0, n + 3):
        x = -5.5 + i * (9 / n)  # from min a bit beyond max
        y = -4.3 + j * (8.05 / n)
        # dist = np.sqrt(x**2 + y**2)
        col = 1 - model.predict(np.array([[x, y]]))[0][0]
        plt.scatter(x, y, color=(col, col, col))

for i, point in enumerate(X):
    col = 'b' if Y[i] == 0 else 'r'
    plt.scatter(point[0], point[1], color=col)

plt.grid()
plt.show()

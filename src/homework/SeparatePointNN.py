from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)

X = np.array([ # x range 8.5, y range 7.8
    [-5.5, 3.5],
    [0.75, 3.75],
    [0.25, 0.4],
    [3.5, 0.4],
    [-0.4, -0.35],
    [-2.3, -4.3],
    [3, -3]
])
Y = np.array([0, 0, 1, 0, 1, 0, 1])  # blue=0, red=1

# radiuses = np.array([np.sqrt(x**2 + y**2) for x, y in X])
# X = np.column_stack((X, radiuses))

model = Sequential()
model.add(Dense(7, input_dim=2, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=500)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

n = 25
for i in range(0, n + 3):
    for j in range(0, n + 3):
        x = -5.5 + i * (9 / n)
        y = -4.3 + j * (8.05 / n)
        # dist = np.sqrt(x**2 + y**2)
        col = 1 - model.predict(np.array([[x, y]]))[0][0]
        plt.scatter(x, y, color=(col, col, col))

for i, point in enumerate(X):
    col = 'b' if Y[i] == 0 else 'r'
    plt.scatter(point[0], point[1], color=col)

plt.grid()
plt.show()

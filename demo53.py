import numpy
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras

dataset1 = numpy.loadtxt('data/demo53_datasets.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

model = Sequential()
print(type(model))
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.05)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())
model.fit(inputList, resultList, epochs=200, batch_size=25)
scores = model.evaluate(inputList, resultList)
print(scores)
print(model.metrics_names)
print(f"{model.metrics_names[0]} => {scores[0]}")
print(f"{model.metrics_names[1]} => {scores[1]}")
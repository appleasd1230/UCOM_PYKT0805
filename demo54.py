import numpy
from keras.layers import Dense
from keras.models import Sequential
import keras

dataset1 = numpy.loadtxt('data/demo53_datasets.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def createModel():
    #global model
    model = Sequential()
    print(type(model))
    model.add(Dense(8, input_dim=8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

MODEL_PATH = 'models/demo54'
model1 = createModel()
model1.fit(inputList, resultList, epochs=200, batch_size=20)
# validate using trained model
scores = model1.evaluate(inputList, resultList)
print(scores)
print(model1.metrics_names)
print(f"{model1.metrics_names[0]} => {scores[0]}")
print(f"{model1.metrics_names[1]} => {scores[1]}")
keras.models.save_model(model1, MODEL_PATH)

print("now with un-trained model")
model2 = createModel()
scores = model2.evaluate(inputList, resultList)
print(scores)
print(model2.metrics_names)
print(f"{model2.metrics_names[0]} => {scores[0]}")
print(f"{model2.metrics_names[1]} => {scores[1]}")

print("load from previous save")
model3 = keras.models.load_model(MODEL_PATH)
scores = model3.evaluate(inputList, resultList)
print(scores)
print(model3.metrics_names)
print(f"{model3.metrics_names[0]} => {scores[0]}")
print(f"{model3.metrics_names[1]} => {scores[1]}")
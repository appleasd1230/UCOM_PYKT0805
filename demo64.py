from keras.datasets import boston_housing
from keras import models, layers

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()
print(train_data.shape, test_data.shape)

# get average
mean = train_data.mean(axis=0)
print(mean)
train_data -= mean
std = train_data.std(axis=0)
print(std)
train_data /= std
print("now apply to test")
test_data -= mean
test_data /= std


def build_model():
    model1 = models.Sequential()
    model1.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model1.add(layers.Dense(64, activation='relu'))
    model1.add(layers.Dense(1))
    model1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model1


model1 = build_model()
print(model1.summary())
model1.fit(train_data, train_target, epochs=100, batch_size=10)

for i,reference in zip(test_data, test_target):
    predict = model1.predict(i.reshape(1, -1))
    print(f"predict={predict[0][0]:.1f}, actual={reference}")
import keras.utils as utils
from keras.utils import np_utils
from tensorflow.python.keras.utils.np_utils import to_categorical

orig = [3, 5, 7, 9, 4]
NUM_DIGITS = 15
to_categorical(orig, NUM_DIGITS)
print(f"before conversion, data={orig}")
converted = utils.to_categorical(orig, NUM_DIGITS)
print(f"after conversion, data={converted}")
converted2 = np_utils.to_categorical(orig, NUM_DIGITS)
print(f"after conversion, data={converted2}")
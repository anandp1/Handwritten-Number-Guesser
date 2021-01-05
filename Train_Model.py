from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Shape of X_train: (60000, 28, 28)
# Shape of y_train: (60000,)
# Shape of X_test: (10000, 28, 28)
# Shape of y_test: (10000,)
# Convert to greyscale
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# Shape of X_train: (60000, 28, 28, 1)
# Shape of y_train: (60000,)
# Shape of X_test: (10000, 28, 28, 1)
# Shape of y_test: (10000,)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Declare the model
model = Sequential()

# Declare the layers
layer_1 = Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1))
layer_2 = Conv2D(64, kernel_size=3, activation="relu")
layer_3 = Flatten()
layer_4 = Dense(10, activation="softmax")

# Add the layers to the model
model.add(layer_1)
model.add(layer_2)
model.add(layer_3)
model.add(layer_4)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

# Save the model
model.save(r'C:\Users\Admin\PycharmProjects\OpenCvProject\myModel.h5')

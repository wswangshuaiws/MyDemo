from keras.datasets import mnist
from keras.layers import Dense,GRU
from keras.utils import to_categorical
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(GRU(units=64,input_shape=(28,28)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=6, batch_size=128, verbose=1)

model.summary()
score = model.evaluate(x_test, y_test,batch_size=128, verbose=1)
print(score)
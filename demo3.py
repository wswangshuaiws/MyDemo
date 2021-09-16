from keras.datasets import mnist
from keras.layers import Dense, LSTM, SimpleRNN,RNN
from keras.utils import to_categorical
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
# input_shape：输入维度，如果 LSTM 作为第一层的话，应该显示的设置该值
# input_dim:输入维度，如果是第一层需要显示的设置该值
# batch_input_shape:(sample,time_step,input_dim)
# units：神经元个数
# return_sequences: 如果为 True 的话，只返回最后一个状态的输出，是一个 2D 张量，否则返回所有序列状态的输出，是一个 3D 张量
#   在单层神经网络中，应该将其设置为 False
#   在多层神经网络中，应该将最后一层设置为 False，其余层设置为 False
model.add(LSTM(units=16, return_sequences=True, batch_input_shape=(None,28,28)))
model.add(LSTM(units=16))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=1)

model.summary()
score = model.evaluate(x_test, y_test,batch_size=128, verbose=1)
print(score)
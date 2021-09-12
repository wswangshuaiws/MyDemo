'''
keras使用举例：使用keras实习了一个全连接深度网络，用于预测手写数字
'''
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils

# 加载数据集,若目录下没有，将会自动下载
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# reshape函数的作用：更改数组形状
# 第一个参数：更改后的行数
# 地二个参数：更改后的列数
X_train = X_train.reshape(len(X_train),-1)
X_test = X_test.reshape(len(X_test),-1)

# to_categorical的作用：将类别向量转换为二进制矩阵（将原来类别向量中的每个值都转换为矩阵里的一个行向量）
# 第一个参数：类别向量
# 第二个参数：行向量的维数
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# 建立序列模型：各层之间是依次顺序的线性关系，模型结构通过一个列表来制定
# keras 中除了序列模型还提供通用模型 Model,可以自定义的东西多，但是更复杂
model = Sequential()

# add() 函数的作用：向网络中添加层
# add() 函数的参数只有一个，表示一层网络
# Dense() 函数实现了全连接层，参数详见函数原型
# input_shape：即张量的形状
# input_dim：代表张量的维度
# relu 和 softmax 均为激活函数
model.add(Dense(input_dim=28*28,units=500,activation='relu'))
for i in range(8):
    model.add(Dense(units=500,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

# compile()函数的作用：定义损失函数，优化器和准确率评估标准
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 开始训练
model.fit(X_train, Y_train, batch_size=500, epochs=3)

# 保存模型
# mp = "model.h5"
# model.save(mp)

# 加载模型
# from keras.models import load_model
# model = load_model("model.h5")

# evaluate() 函数的作用：输入数据和标签，返回损失和目标值
# 第一个结果代表 loss
# 第二个结果代表 选定的目标值，比如准确率
result = model.evaluate(X_test,Y_test)
print('Total loss on testing set:',result[0])
print('Accuracy of testing set:',result[1])

# predict() 函数的作用：输入测试数据,输出预测结果
y_pred = model.predict(X_test)
print(y_pred)
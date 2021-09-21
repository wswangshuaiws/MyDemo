import numpy
from PIL import Image
from keras.datasets import mnist
from keras.layers import Dense, GRU
from keras.utils import to_categorical
from keras.models import Sequential

# 加载数据集,包括训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 更改数据集中 label 的形状
# 训练集：(60000)  转换为   (60000,10)
# 测试集：(10000)  转换为   (10000,10)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 建立模型：网络结构为：
#   GRN：卷积核的数目为64,采用 sigmoid 作为激活函数
#   全连接层：神经元的数目为10，采用softmax作为激活函数，输出可能数字的最大概率
model = Sequential()
model.add(GRU(units=64, input_shape=(28, 28)))
model.add(Dense(10, activation='softmax'))

# 配置优化器、损失函数和准确率评测标准
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 使用训练集开始训练
# 每次取128个并行计算
# 对整个数据集训练10次
model.fit(x_train, y_train, epochs=6, batch_size=128, verbose=1)

# 评估模型
# evaluate() 方法返回两个结果，第一个为损失;第二个为准确率
score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('loss:', score[0])
print('accuracy:', score[1])

# 对结果进行预测，这里对测试集中第一张图片进行预测
# predict 的返回值是一个个形状为(1,10)的矩阵，通过对其进行遍历，获取最大值所在位置的索引，即为预测结果
res = model.predict(x_test[0 : 1, 0:, 0 :, 0 :])
max = res[0][0]
index = 0
for i in range(1, 10):
    if res[0][i] > max:
        max = res[0][i]
        index = i
print('预测结果为', index)

# 输出上一步骤预测的图片，方便对比结果
img = x_test[0]
img = img.reshape(28, 28)
pil_img = Image.fromarray(numpy.uint8(img))
pil_img.show()

# 保存模型
mp = "model.h5"
model.save(mp)

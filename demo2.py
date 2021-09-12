'''
CNN 的简单使用
'''
import keras
import numpy
from keras import Sequential
from keras.datasets import mnist
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import np_utils
from PIL import Image

# 加载数据集,包括训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 更改数据集中 image 的形状
# keras 底层默认使用 Tensforflow
# 而 Tensforflow 使用的数据格式是channels_last即：（样本数，行数（高），列数（宽），通道数）
# 训练集：(60000,28,28)  转换为  (60000,28,28,1)
# 测试集：(10000,28,28)  转换为  (10000,28,28,1)
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

# 更改数据集中 label 的形状
# 训练集：(60000)  转换为   (60000,10)
# 测试集：(10000)  转换为   (10000,10)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 建立模型：网络结构为：
#   卷积层：卷积核的数目为32，卷积核的大小为(3,3),采用relu作为激活函数
#   最大池化层：核化池的尺寸为(2,2)
#   全连接层：神经元的数目为128，采用relu作为激活函数
#   全连接层：神经元的数目为128，采用relu作为激活函数
#   全连接层：神经元的数目为10，采用softmax作为激活函数，输出可能数字的最大概率
model = Sequential()
model.add(Conv2D(32, (3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# 由于测试时发现，虽然训练集准确度很高，但是测试集准确度不理想的情况，因此考虑采用 Dropout 提高测试的准确度
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 配置优化器、损失函数和准确率评测标准
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 使用训练集开始训练
# 每次取128个并行计算
# 对整个数据集训练10次
model.fit(x_train, y_train,
          batch_size = 128,
          epochs=10)

# 保存模型
mp = "model.h5"
model.save(mp)

# 加载模型
model = load_model("model.h5")

# 评估模型
# evaluate() 方法返回两个结果，第一个为损失;第二个为准确率
score = model.evaluate(x_test, y_test)
print('loss:',score[0])
print('accuracy',score[1])

# 对结果进行预测，这里对测试集中第一张图片进行预测
# predict 的返回值是一个个形状为(1,10)的矩阵，通过对其进行遍历，获取最大值所在位置的索引，即为预测结果
res = model.predict(x_test[0:1,0:,0:,0:])
max = res[0][0]
index = 0
for i in range(1,10):
    if res[0][i] > max:
        max = res[0][i]
        index =i
print('预测结果为',index)

# 输出上一步骤预测的图片，方便对比结果
img = x_test[0]
img = img.reshape(28, 28)
pil_img = Image.fromarray(numpy .uint8(img))
pil_img.show()
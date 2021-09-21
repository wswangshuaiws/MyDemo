from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc


def ReadFromTxt(fileName, flag=1, encoding='utf-8'):
    '''
    在指定文本文件中读取二维矩阵，要求二维矩阵不能有空行，列于列之间用空格或者制表符间隔
    :param fileName: 文件路径
    :param flag: 数据格式，目前可选 1，2,默认值为1
        取 1 的时候，将从文件中按照整形数据的格式读取数据
        取 2 的时候，将从文件中按照浮点数的格式读取数据
    :param encoding: 文件编码方式，默认采用 utf-8
    :return: 读取得到的二维矩阵
    '''
    matrix = []
    try:
        with open(fileName, encoding=encoding) as file_obj:
            while True:
                matrixRow = []
                # 在文件中读取一行数据
                content = file_obj.readline()
                # 判断数据是否为空，若为空将退出循环
                if not content:
                    break
                # 将读取得到的数据，存储进一个列表
                i = 0
                while i < len(content):
                    temp = content[i]
                    i = i + 1
                    while i < len(content) and content[i] != ' ' and content[i] != '\t':
                        temp = temp + content[i]
                        i = i + 1
                    if flag == 1:
                        matrixRow.append(int(temp))
                    elif flag == 2:
                        matrixRow.append(float(temp))
                    while i < len(content) and content[i] == ' ' and content[i] != '\t':
                        i = i + 1
                # 将新生成的一行列表，添加进矩阵
                matrix.append(matrixRow)
    except FileNotFoundError:
        print(f'{fileName} 这个文件不存在！')
    return np.array(matrix)


def create_model():
    '''
    创建 CNN 模型，模型结构为
        卷积层：卷积核的大小是（2，2），神经元数目是16，使用 relu 激活函数
        最大池化层：核化池的尺寸为(1,2)
        卷积层：卷积核的大小是（1，3），神经元数目是32，使用 relu 激活函数
        最大池化层：核化池的尺寸为(1,2)
        全连接层：神经元数目是32，使用 relu 激活函数
        全连接层：神经元数目是2，使用 softmax 激活函数
    :return: 返回创建完成的神经网络
    '''
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(2, 1140, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(64, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


def curve(FPR, TPR, P):
    '''
    绘制 ROC 和 PR 曲线，并给出 ROC 和 PR 曲线的面积
    :param FPR: 正确识别的正例，所占正例的比例
    :param TPR: 误认为正例的反例，所占反例的比例
    :param P: 表示正确识别的正例，占所有被预测为正例的比例
    '''
    plt.figure()
    plt.subplot(121)
    plt.xlim(0.0, 1.0)
    plt.ylim(00.0, 1.)
    plt.title("ROC curve  (AUC = %.4f)" % (auc(FPR, TPR)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(FPR, TPR)

    plt.subplot(122)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 0.5)
    plt.title("PR curve  (AUPR = %.4f)" % (auc(TPR, P) + (TPR[0] * P[0])))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(TPR, P)
    plt.show()


def create_and_evaluate_model():
    '''
    对数据进行五倍交叉验证
    '''
    # 从文件读入数据
    # 读入 IncRNA 与疾病关联矩阵
    lnc_dis_association = ReadFromTxt("lnc_dis_association.txt")
    # 读入疾病相似性矩阵
    dis_sim_matrix_process = ReadFromTxt("dis_sim_matrix_process.txt", 2)
    # 读入 IncRNA 功能相似性矩阵
    lnc_sim = ReadFromTxt("lnc_sim.txt", 2)
    # 读入 miRNA 和疾病关联矩阵
    mi_dis = ReadFromTxt("mi_dis.txt")
    # 读入 IncRNA 和 miRNA 互作矩阵
    yuguoxian_lnc_mi = ReadFromTxt("yuguoxian_lnc_mi.txt", 2)

    # 以下两个列表，用于记录 IncRNA 和疾病关联矩阵中，有关联和没有关联的位置
    localOfTrue = []
    localOfFalse = []

    # 遍历IncRNA 和疾病关联矩阵，以确定 localOfTrue 和 localOfFalse 的值
    for i in range(lnc_dis_association.shape[0]):
        for j in range(lnc_dis_association.shape[1]):
            # 如果有关联，则将该点的坐标（x，y），放入 localOfTrue
            if lnc_dis_association[i][j] == 0:
                localOfFalse.append((i, j))
            # 如果没有关联，则将该点的坐标（x，y），放入 localOfFalse
            else:
                localOfTrue.append((i, j))

    # 随机打乱 localOfTrue 和 localOfFalse 列表
    shuffle_ix = np.random.permutation(np.arange(len(localOfTrue)))
    localOfTrue = np.array(localOfTrue)[shuffle_ix]
    shuffle_ix = np.random.permutation(np.arange(len(localOfFalse)))
    localOfFalse = np.array(localOfFalse)[shuffle_ix]

    # 代表有关联的节点组成的集合（由于以上已经将该 localOfTrue 打乱，因此集合的顺序是随机的），x_true 代表 image,y_true代表 label
    # x_true 的形状是（5，537，2，1140，1）
    # y_true 的形状是（5，337，2）
    x_true = []
    y_true = []

    # 有关联的节点会被分成5份，localOfTrueSplit用于存储分割后的结果
    # 其形状为 （5，337，2），分别代表 5 份数据，一份数据有537 个，每个数据是一个（x,y）的形状
    localOfTrueSplit = []

    # 计算 localOfTrueSplit,x_true 和 y_true,
    # 由于数据集中只存在 2687 个正例，因此分成五分的结果是：每份537个
    for i in range(5):
        localOfTrueSplit.append(localOfTrue[i * 537:(i + 1) * 537])

        image = []
        label = []

        for j in range(len(localOfTrueSplit[i])):
            k, l = localOfTrueSplit[i][j]
            A1 = lnc_sim[k:k + 1, 0:]
            A2 = lnc_dis_association[k:k + 1, 0:]
            A3 = yuguoxian_lnc_mi[k:k + 1, 0:]
            B1 = lnc_dis_association[0:, l:l + 1].reshape(1, -1)
            B2 = dis_sim_matrix_process[k:k + 1, 0:]
            B3 = mi_dis[0:, l:l + 1].reshape(1, -1)
            image.append(np.r_[np.c_[A1, A2, A3], np.c_[B1, B2, B3]].reshape(2, 1140, 1))
            label.append(np.array([0, 1]))

        x_true.append(image)
        y_true.append(label)

    x_true = np.array(x_true)
    y_true = np.array(y_true)

    # 由于正例只有 2687 个，因此我们随机取 2687 个，并分成 5 份
    # 代表没有关联的节点组成的集合（由于以上已经将该 localOfFalse 打乱，因此集合的顺序是随机的），x_false 代表 image,y_false代表 label
    # x_false 的形状是（5，537，2，1140，1）
    # y_false 的形状是（5，337，2）
    x_false = []
    y_flase = []

    # 有关联的节点会被分成5份，localOfFalseSplit用于存储分割后的结果
    # 其形状为 （5，337，2），分别代表 5 份数据，一份数据有537 个，每个数据是一个（x,y）的形状
    localOfFalseSplit = []

    # 计算 localOfTrueSplit,x_true 和 y_true
    for i in range(5):
        localOfFalseSplit.append(localOfFalse[i * 537:(i + 1) * 537])
        image = []
        label = []
        for j in range(len(localOfFalseSplit[i])):
            k, l = localOfFalseSplit[i][j]
            A1 = lnc_sim[k:k + 1, 0:]
            A2 = lnc_dis_association[k:k + 1, 0:]
            A3 = yuguoxian_lnc_mi[k:k + 1, 0:]
            B1 = lnc_dis_association[0:, l:l + 1].reshape(1, -1)
            B2 = dis_sim_matrix_process[k:k + 1, 0:]
            B3 = mi_dis[0:, l:l + 1].reshape(1, -1)

            image.append(np.r_[np.c_[A1, A2, A3], np.c_[B1, B2, B3]].reshape(2, 1140, 1))
            label.append(np.array([1, 0]))

        x_false.append(image)
        y_flase.append(label)

    x_false = np.array(x_false)
    y_flase = np.array(y_flase)

    # 将整个数据集，做成神经网络要求的输入，后续将使用训练好的模型来预测该矩阵，以获得得分矩阵
    x_all = []

    # 将整个数据集中，IncRNA 和疾病的关联关系，存储进去，后续将用其和得分矩阵进行比较，以计算参数
    y_all = []

    # 计算 x_all 和 y_all
    for k in range(lnc_dis_association.shape[0]):

        temp1 = []
        temp2 = []

        for l in range(lnc_dis_association.shape[1]):
            A1 = lnc_sim[k:k + 1, 0:]
            A2 = lnc_dis_association[k:k + 1, 0:]
            A3 = yuguoxian_lnc_mi[k:k + 1, 0:]
            B1 = lnc_dis_association[0:, l:l + 1].reshape(1, -1)
            B2 = dis_sim_matrix_process[k:k + 1, 0:]
            B3 = mi_dis[0:, l:l + 1].reshape(1, -1)
            temp1.append(np.r_[np.c_[A1, A2, A3], np.c_[B1, B2, B3]].reshape(1, 2, 1140, 1))
            temp2.append(lnc_dis_association[k][l])

        x_all.append(temp1)
        y_all.append(temp2)

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    # 用于存储五倍交叉验证中，计算得到的 FPR ，TPR 和P 的值
    FPRs = []
    TPRs = []
    Ps = []

    # 五倍交叉验证
    for i in range(5):

        # 创建训练集
        x_train = []
        y_train = []
        localOfTrueTrain = []

        # 在数据集中加入 4 份正例以及 4 份反例，并记住所加入正例的位置，方便在计算得分矩阵的时候，将其剔除
        for j in range(5):
            if i != j:
                x_train.append(x_true[j])
                x_train.append(x_false[j])
                y_train.append(y_true[j])
                y_train.append(y_flase[j])
                localOfTrueTrain.append(localOfTrueSplit[j])

        # 训练集四份正例和四份反例，因此长度是 537 * 8 = 4296
        x_train = np.array(x_train)
        x_train = x_train.reshape(4296, 2, 1140, 1)

        y_train = np.array(y_train)
        y_train = y_train.reshape(4296, 2)

        # 我们只需记住四份正例的位置，四份正例共 537 * 4 =2148
        localOfTrueTrain = np.array(localOfTrueTrain)
        localOfTrueTrain = localOfTrueTrain.reshape(2148, 2)

        # 因为以上是按照一定规律在训练集中加入数据，因此这一步需要将数据打散
        # 采用相同的 shuffle_ix 可以保证 image 和 label 按照同样的顺序被打散
        shuffle_ix = np.random.permutation(np.arange(len(x_train)))
        x_train = x_train[shuffle_ix]
        y_train = y_train[shuffle_ix]

        # 创建模型，并定义损失函数和优化器
        modelTemp = create_model()
        modelTemp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 开始训练
        modelTemp.fit(x_train, y_train, batch_size=128, epochs=5)

        # 将此次训练所使用的正例的位置填入负一，代表不计算他的分数
        res = np.full((lnc_dis_association.shape[0], lnc_dis_association.shape[1]), 0., dtype=float)
        for k in range(len(localOfTrueTrain)):
            row, col = localOfTrueTrain[k]
            res[row, col] = -1

        # 计算得分矩阵
        for k in range(lnc_dis_association.shape[0]):
            for l in range(lnc_dis_association.shape[0]):
                if res[k][l] != -1:
                    res[k][l] = modelTemp.predict(x_all[k][l])[0][1]

        # 将得分矩阵的每行从大到小排序，并记住原来的位置
        old_id = np.argsort(-res)
        res = -res
        res.sort()
        res = -res

        # 计算得分矩阵每行中有效数据的个数
        f = []
        for k in range(lnc_dis_association.shape[0]):
            temp = 0
            for l in range(lnc_dis_association.shape[1]):
                if res[k][l] != -1:
                    temp = temp +1
            f.append(temp)

        # 计算得分矩阵中每行中有效数据个数的最小值
        min_f = min(f)

        # 表示经过裁剪后的得分矩阵（每一行的长度都一样）
        resMin =np.full((lnc_dis_association.shape[0], min_f), 0., dtype=float)

        # 表示经过裁剪后，每个元素与未排序前的对应关系
        new_id = np.full((lnc_dis_association.shape[0], min_f), 0., dtype=int)

        # 将得分矩阵每一行裁剪成相同的形状
        for k in range(lnc_dis_association.shape[0]):
            for l in range(min_f):
                resMin[k][l] = res[k][int(np.round_((f[k] / min_f)*(l+1))) - 1]
                new_id[k][l] = old_id[k][int(np.round_((f[k] / min_f)*(l+1)))- 1]

        # 求解每一列平均的 FPR TPR 和 P
        # 然后分别将每一列的结果加入列表
        FPR = []
        TPR = []
        P = []

        for col in range(min_f):
            sum = 0
            FPRCol = np.full(200, 0, dtype=float)
            TPRCol = np.full(200, 0, dtype=float)
            PCol = np.full(200, 0, dtype=float)

            # 我们采用的是每隔0.05 取一个采样点，因此（0，1）共取 200 个点
            for k in np.arange(0, 1, 0.05):

                # 统计每一列中 TP ，FN，FP和TN的总数
                TPRol = 0
                FNRol = 0
                FPRol = 0
                TNRol = 0

                for row in range(240):
                    if resMin[row][col] > k and y_all[row][new_id[row][col]] ==1:
                        TPRol = TPRol + 1
                    elif resMin[row][col] < k and y_all[row][new_id[row][col]] ==1:
                        FNRol = FNRol + 1
                    elif resMin[row][col] > k and y_all[row][new_id[row][col]] ==0:
                        FPRol = FPRol + 1
                    else:
                        TNRol = TNRol + 1

                # 实验中发现，有可能在某一列中会存在 FP 和 TN 均为0的情况，因此对其做出标记，方便后续计算
                if FPRol + TNRol == 0:
                    FPRCol[sum] = -1
                else:
                    FPRCol[sum] = FPRol / (FPRol + TNRol)
                if TPRol + FNRol == 0:
                    TPRCol[sum] = -1
                else:
                    TPRCol[sum] = TPRol / (TPRol + FNRol)
                if TPRol + FPRol == 0:
                    PCol[sum] = -1
                else:
                    PCol[sum] = TPRol / (TPRol + FPRol)

                sum = sum + 1

            # 将每一列的结果加入列表
            FPR.append(FPRCol)
            TPR.append(TPRCol)
            P.append(PCol)

        # 将该次交叉验证得到的结果存储进列表
        FPRs.append(FPR)
        TPRs.append(TPR)
        Ps.append(P)

    # 计算五次交叉验证后，得到的 FPR，TPR和P的个数
    f = []
    for k in range(5):
        f.append(len(FPRs[k]))

    # 求个数的最小值
    min_f = min(f)

    # 计算将FPR，TPR和P 转换为相同长度后的结果（切割掉多余数据）
    FPRMin = np.full((5, min_f), 0., dtype=list)
    TPRMin = np.full((5, min_f), 0., dtype=list)
    PMin = np.full((5, min_f), 0., dtype=list)

    for k in range(5):
        for l in range(min_f):
            FPRMin[k][l] = FPRs[k][int(np.round_((f[k] / min_f)) * (l + 1) - 1)]
            TPRMin[k][l] = TPRs[k][int(np.round_((f[k] / min_f)* (l + 1))  - 1)]
            PMin[k][l] = Ps[k][int(np.round_((f[k] / min_f)* (l + 1))  - 1)]

    # 将以上五次的结果平均，计算最终的 FPR，TPR 和 P
    FPR = []
    TPR = []
    P = []

    # 我们采用的是每隔0.05 取一个采样点，因此（0，1）共取 200 个点
    for m in range(200):
        tempFPR = 0
        tempTPR =0
        tempP =0

        # 存储FPR，TPR 和 P等于 0 的个数
        fprsum = 0
        tprsum = 0
        psum =0

        for k in range(5):
            for l in range(min_f):
                if PMin[k][l][m] != -1:
                    tempP = tempP + PMin[k][l][m]
                else:
                    psum = psum +1
                if FPRMin[k][l][m] != -1:
                    tempFPR = tempFPR + FPRMin[k][l][m]
                else:
                    fprsum = fprsum + 1
                if TPRMin[k][l][m] != -1:
                    tempTPR = tempTPR + TPRMin[k][l][m]
                else:
                    tprsum = tprsum + 1

        FPR.append(tempFPR / (5 * min_f - fprsum))
        TPR.append(tempTPR / (5 * min_f - tprsum))
        P.append(tempP / (5 * min_f - psum))

    # 绘制 PRC 和 PR 曲线
    curve(FPR,TPR,P)


if __name__ == "__main__":
    create_and_evaluate_model()
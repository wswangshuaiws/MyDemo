'''
文件读取工具类
'''


class FileTool():
    @staticmethod
    def ReadFromTxt(fileName, flag=1, encoding='utf-8'):
        '''
        在指定文本文件中读取二维矩阵，要求二维矩阵不能有空行，列于列之间用空格 或者 制表符间隔
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
        return matrix

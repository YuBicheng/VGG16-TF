import os

def getextension(filename):
    """
    获取文件的拓展名
    :param filename:
    :return: 扩展名
    """
    (filepath,tempfilename) = os.path.split(filename);
    (shotname,extension) = os.path.splitext(tempfilename);
    return extension

def make_train_txt(path, label, writer):
    for d in os.scandir(path):
        if getextension(d)=='.jpg':
            writer.write(os.path.join(path, d.name) + ' ' + str(label) + '\n')

def create_trainlist(root_path, name):
    """
    生成数据集的图片地址和标签txt
    格式为
    图片1地址 标签
    图片2地址 标签
    ···
    :param root_path: 图片数据根目录
    :return:无返回值
    """
    #创建标签映射文件，行数就是对应标签例如daisy在第0行 标签就为0
    class_list = []
    with open("./" + name + '.names', 'w') as f:
        for dir in os.scandir(root_path):
            if dir.is_dir():
                class_list.append(dir.name)
                f.write(dir.name + '\n')
    with open("./train.txt", 'w') as g:
        for i in range(len(class_list)):
            make_train_txt(os.path.join(root_path, class_list[i]), i, g)

if __name__=='__main__':
    create_trainlist('./flower_photos', 'flower')
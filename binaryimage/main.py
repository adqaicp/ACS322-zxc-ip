import os
from PIL import Image

# 源目录
MyPath = 'C:\\Users\\adqaicp\\Desktop\\HWDB1.0\\00091\\'
# 输出目录
OutPath = 'C:\\Users\\adqaicp\\Desktop\\HWDB1.0\\00091-b\\'


def processImage(filesoure, destsoure, name, imgtype):
    '''
    filesoure是存放待转换图片的目录
    destsoure是存在输出转换后图片的目录
    name是文件名
    imgtype是文件类型
    '''
    imgtype = 'bmp' if imgtype == '.bmp' else 'png'
    # 打开图片
    im = Image.open(filesoure + name)
    # =============================================================================
    #     #缩放比例
    #     rate =max(im.size[0]/640.0 if im.size[0] > 60 else 0, im.size[1]/1136.0 if im.size[1] > 1136 else 0)
    #     if rate:
    #         im.thumbnail((im.size[0]/rate, im.size[1]/rate))
    # =============================================================================

    img = im.convert("RGB")
    pixdata = img.load()
    # 二值化
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y][0] < 60:
                pixdata[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y][1] < 200:
                pixdata[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y][2] > 0:
                pixdata[x, y] = (255, 255, 255, 255)
    img.save(destsoure + name, imgtype)


def run():
    # 切换到源目录，遍历源目录下所有图片
    os.chdir(MyPath)
    for i in os.listdir(os.getcwd()):
        # 检查后缀
        postfix = os.path.splitext(i)[1]
        if postfix == '.bmp' or postfix == '.png':
            processImage(MyPath, OutPath, i, postfix)


if __name__ == '__main__':
    run()

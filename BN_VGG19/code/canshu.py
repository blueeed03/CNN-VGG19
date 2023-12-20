"""路径参数"""
train_img_dir = "C:/Users/bluuue/Desktop/CNN_newlogo/sample1/Logo-2K+/Accessories" #训练集路径
test_img_dir = "C:/Users/bluuue/Desktop/CNN_newlogo/sample1/Logo-2K+/Accessories" #测试集路径
model_save_dir = "C:/Users/bluuue/Desktop/CNN_newlogo/savemodel/modelsaved_model" #模型保存路径


"""种类参数"""
labels = [
'Benz',
'Buick',
'Citroen' ,
'FAW',
'Fukude',
'Honda',
'Hyundai' ,
'KIA',
'Lexus',
'Mazda',
'MG',
'Nissan',
'Toyota',
'Volkswagen' ,
'AoDi',
'BaoJun',
'BaoMa',
'BaoShiJie',
'BeiQiHuanSu',
'BeiQiWeiWang',
'BeiQiXinNengYuan',
'BenTeng',
'ChuanQi',
'DongFeng',
'DongNan',
'FengTianHuangGuan',
'JiLi',
'JiLiDiHao',
'JiLiQuanQiuYing',
'JiLiYingLun',
'JiPu',
'KaiRui',
'KaiDiLaKe',
'LiFan',
'QiRui',
'QiChen',
'SiBaLu',
'WeiLin',
'ZhongHua',
'ZhongTai']#种类标签
num = len(labels) #种类数


"""图片参数"""
image_width = 64 #图片统一宽度
image_height = 64 #图片统一高度


"""训练参数"""
train_ratio = 0.9 #训练集比例
train_batch_size = 2 #训练集批次大小
val_batch_size = 2 #验证集批次大小
epochs = 20 #训练次数
learning_rate = 0.0001 #学习率  0.000005





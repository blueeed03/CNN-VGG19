# -*- coding: utf-8 -*-
# @Time     :2021/12/6 16:39
# @File     :canshu.py
# @Software :PyCharm
# @Project  :参数信息
# @Content  :可变参数信息


"""路径参数"""
train_img_dir = "C:/Users/bluuue/Desktop/CNN_newlogo/sample1/Logo-2K+/Accessories" #训练集路径
test_img_dir = "C:/Users/bluuue/Desktop/CNN_newlogo/sample1/Logo-2K+/Accessories" #测试集路径
model_save_dir = "C:/Users/bluuue/Desktop/CNN_newlogo/savemodel/modelsaved_model" #模型保存路径


"""种类参数"""
labels = [
'24seven', 'Ace', 'aceh', 'Admiral', 'Aetrex', 'Agio', 'Aim', 'Akubra', 'ALAN', 'Alexia', 'Allstar', 'ara', 'Artemide', 'Ashbury', 'Atlantic', 'Atlantic Airlines', 'Audar', 'Aurora', 'Avanti', 'Baba', 'BACCARAT', 'BAE', 'Baker Skateboards', 'Balance', 'Bandit', 'Barcelo', 'Basso', 'Beeline', 'Berghaus', 'Berkshire', 'Bexley', 'Black and Red', 'Black Label Skateboards', 'Bobdog', 'Bona', "Boscov's", 'BOSS', 'Boxfresh', 'Braun', 'Bravo', 'Breguet', 'BridgePort', 'Brooke', 'Brooklyn', 'BT', 'Butler', 'Buxton', 'C&M Airways', 'Cabana', 'Calypso', 'Carhartt', 'Cartier', 'Casa Blanca', 'Casino', 'Centurion', 'CHRISTOFLE', "Church's", 'CMC', 'Colt', 'Consort', 'Converse', 'Cresta', 'Crown', 'Crumpler', 'Crystal', 'Dansko', 'DC Shoes', 'diamond', 'Dignity', 'Domestos', 'Domino', 'Domus', 'Don Jose', 'Dor', 'Drifter', 'DUX', 'DVS Shoes', 'DY', 'Eagle Creek', 'Eastpak', 'Ecco', 'Elbe', 'Emerica', 'Emo', 'Estrella', 'Etnies', 'Excelsior', 'Georgia Boot', 'Goya', 'Hamilton', 'Hammary', 'Hanwag', 'Healing', 'Heritage', 'HMT', 'Hoffman', 'Howard', 'Imperia', 'Jensen', 'Joy', 'Keen', 'Kendo', 'Klogs', 'Krystal', 'KYOCERA', 'Lane Furniture', 'Leaf', 'Lexington', 'Libbys', 'Lloyd', "LLoyd's", 'Longines', 'Look', 'Love is', 'Lowepro', 'Makro', 'Manhattan Portage', 'MARC BY MARC JACOBS', 'Margherita', 'Matchmakers', 'Matt', 'Maverik Inc', 'MBT', 'Meijer', 'Meindl', 'Memphis', 'Mendocino', 'miffy', 'Moca', 'Monopoly', 'Murray', 'NEI', 'Nelson', 'Nerds', 'Nike', 'Nike SB', 'Nippon', 'Nordstrom Rack', "O'Neill", 'OMAS', 'ONE', 'Origen', 'Osiris Shoes', 'PacSun', 'Palladium', 'Panama', 'Papillio', 'Patio', 'Patron', 'PG', 'Phillies', 'Pirates', 'Premium', 'Primo', 'Proto', 'puma', 'Quiksilver', 'RaceWay', 'Realtor', 'RedDragnfly', 'Redline', 'Riley', 'Rodenstock', 'Rogue', 'Romika', 'Sennheiser', 'Seven', 'Shake Junt', 'Sigma', 'Skechers', 'Smash', 'Spider-Man', 'SR', 'Steel', 'Sterling', 'Summit', 'Sunlight', 'Sunrise', 'Supreme', 'swan', 'TAG Heuer', 'Tandy', 'Tango', 'Tata', 'Tatiana', 'Teenage Mutant Ninja Turtles', 'The Simpsons', 'Thomas', 'Thorlos', 'TIGI', 'Time', 'Titus', 'Tombstone', 'Trebor', 'Tretorn', 'Trio', 'Trix', 'U.S. Traveler', 'Unox', 'Vanessa', 'Vargas', 'Vasque', 'Vault', 'VH', 'VIA', 'WESCO', 'Wesson', 'Weston', 'Xtep', 'Zamberlan']#种类标签
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





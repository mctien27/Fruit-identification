import numpy as np #import thư viện numpy để làm việc với các mảng đa chiều.
from keras_preprocessing import image #import thư viện keras_preprocessing để xử lý hình ảnh.
import cv2 #import thư viện cv2 để xử lý hình ảnh.
import os #import thư viện os để làm việc với các tệp và thư mục.
import tensorflow as tf #import thư viện tensorflow để xây dựng mạng nơ-ron.
import time #import thư viện time để đo thời gian chạy của chương trình.

#import ImageDataGenerator từ keras_preprocessing để tạo dữ liệu đầu vào cho mạng nơ-ron.
from keras_preprocessing.image import ImageDataGenerator

#tạo một đối tượng ImageDataGenerator để tạo dữ liệu đầu vào cho quá trình huấn luyện mạng nơ-ron.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#tạo một tập dữ liệu huấn luyện từ các hình ảnh trong thư mục 'train' và 
#sử dụng ImageDataGenerator để tạo dữ liệu đầu vào cho mạng nơ-ron.
training_set = train_datagen.flow_from_directory('C:/Users/Admin/Desktop/CNN_Nhom17/CNN_Nhom17/train',
                                                 target_size = (224, 224),
                                                 batch_size = 70,
                                                 class_mode = 'categorical')

#tạo một đối tượng ImageDataGenerator để tạo dữ liệu đầu vào cho quá trình kiểm tra mạng nơ-ron.
test_datagen = ImageDataGenerator(rescale = 1./255)

#tạo một tập dữ liệu kiểm tra từ các hình ảnh trong thư mục 'test' và 
#sử dụng ImageDataGenerator để tạo dữ liệu đầu vào cho mạng nơ-ron.
test_set = test_datagen.flow_from_directory('C:/Users/Admin/Desktop/CNN_Nhom17/CNN_Nhom17/test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')

#tạo một danh sách các lớp (tên các loại rau quả) để sử dụng trong quá trình phân loại.
classes = ['apple','Bnana','beetroot','bell pepper','cabbage','capsicum', 'carrot',
'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans'
'spinach', 'sweetcom', 'sweetpotato', 'tomato', 'turnip', 'watermelon' ]
print("Image Processing.......Compleated")

#tạo một mô hình mạng nơ-ron tuần tự.
cnn = tf.keras.models.Sequential()
print("Building Neural Network.....")

#thêm một lớp tích chập với 32 bộ lọc, kích thước kernel là 3x3, hàm kích hoạt là relu và kích thước đầu vào là 224x224x3.
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[224, 224, 3]))

#thêm một lớp giảm kích thước với kích thước pool là 2x2 và bước là 2.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#thêm một lớp tích chập khác với 32 bộ lọc, kích thước kernel là 3x3 và hàm kích hoạt là relu.
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# thêm một lớp giảm kích thước khác bằng cách lấy giá trị lớn nhất trong mỗi khối 2x2.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#thêm một lớp phẳng hóa để chuyển đổi đầu vào thành một vector 1 chiều.
cnn.add(tf.keras.layers.Flatten())

#thêm một lớp kết nối đầy đủ với 32 đơn vị và hàm kích hoạt là relu.
cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))

#thêm một lớp kết nối đầy đủ khác với 64 đơn vị và hàm kích hoạt là relu.
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))

#thêm một lớp kết nối đầy đủ khác với 128 đơn vị và hàm kích hoạt là relu.
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#thêm một lớp kết nối đầy đủ khác với 256 đơn vị và hàm kích hoạt là relu.
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))

#thêm một lớp kết nối đầy đủ khác với 256 đơn vị và hàm kích hoạt là relu.
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))

#thêm một lớp kết nối đầy đủ cuối cùng với 36 đơn vị (số lượng lớp) và hàm kích hoạt 
#là softmax để tính xác suất cho mỗi lớp.
cnn.add(tf.keras.layers.Dense(units=36, activation='softmax'))

#biên dịch mô hình với thuật toán tối ưu hóa adam, hàm mất mát là categorical_crossentropy và độ đo là accuracy.
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print("Training cnn")

#huấn luyện mô hình với dữ liệu huấn luyện và kiểm tra trong 50 epochs.
cnn.fit(x = training_set, validation_data = test_set, epochs = 100)

cnn.save("model.h7") #lưu mô hình đã huấn luyện vào tệp "model.h7".



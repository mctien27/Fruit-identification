import numpy as np #import thư viện numpy và đặt tên viết tắt là np.
from keras_preprocessing import image #import module image từ thư viện keras_preprocessing.
import cv2 #import thư viện OpenCV.
import os #import thư viện os để thao tác với hệ thống tệp tin.
from tensorflow.keras.models import load_model #import hàm load_model từ thư viện tensorflow.keras.models.

#khởi tạo đối tượng VideoCapture để kết nối với camera.
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")

i = 0 #khởi tạo biến i bằng 0 để đánh số các tệp ảnh lưu trữ.

#khởi tạo danh sách các loại rau củ quả.
classes = ['apple','Bnana','beetroot','bell pepper','cabbage','capsicum', 'carrot',
'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans'
'spinach', 'sweetcom', 'sweetpotato', 'tomato', 'turnip', 'watermelon' ]

#tải mô hình đã được huấn luyện trước đó từ tệp 'model.h7'.
new_model = load_model('model.h7')

#bắt đầu vòng lặp vô hạn để lấy các khung hình từ camera.
while(True):
    r, frame = vid.read() #đọc một khung hình từ camera và lưu vào biến frame.
    cv2.imshow('frame', frame) #hiển thị khung hình đọc được lên màn hình.

    #lưu khung hình vào thư mục với tên tệp là 'final' + số thứ tự i + '.jpg'.
    cv2.imwrite('C:/Users/Admin/Desktop/CNN_Nhom17/CNN_Nhom17/final' + str(i) + ".jpg", frame)

    #tải ảnh từ tệp đã lưu trữ và chuyển đổi kích thước ảnh thành (224, 224).
    test_image = image.load_img('C:/Users/Admin/Desktop/CNN_Nhom17/CNN_Nhom17/final' + str(i) + ".jpg", target_size=(224, 224))
    
    #chuyển đổi ảnh thành một mảng numpy.
    test_image = image.img_to_array(test_image)

    #thêm một chiều mới vào mảng numpy để phù hợp với đầu vào của mô hình.
    test_image = np.expand_dims(test_image, axis=0)

    #sử dụng mô hình để dự đoán loại rau củ quả trong ảnh.
    result = new_model.predict(test_image)

    #lấy kết quả dự đoán đầu tiên từ mảng kết quả.
    result1 = result[0]

    #duyệt qua 36 phần tử đầu tiên của mảng kết quả.
    for y in range(35):
        if result1[y] == 1.: #nếu phần tử thứ y trong mảng kết quả bằng 1.0.
            break
    prediction = classes[y] #lấy tên của loại rau củ quả được dự đoán.
    print(prediction)
    os.remove('C:/Users/Admin/Desktop/CNN_Nhom17/CNN_Nhom17/final' + str(i) + ".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'): # thoát nếu người dùng nhấn phím 'q'.
        break
vid.release() #giải phóng đối tượng VideoCapture.
cv2.destroyAllWindows() #đóng tất cả các cửa sổ hiển thị.
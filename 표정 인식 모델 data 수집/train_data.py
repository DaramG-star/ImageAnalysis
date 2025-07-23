# train_cnn_model.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ⚙️ 설정
img_size = 224
batch_size = 32
num_classes = 5  # 너의 표정 클래스 수

# 📂 폴더 구조 예시
# data/
#   ├── laugh/
#   ├── yawn/
#   ├── serious/
#   ├── surprise/
#   └── ugly/

data_dir = 'data'

# 🌀 데이터 전처리
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 🧠 모델 정의
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 📈 학습
model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# 💾 모델 저장
model.save('expression_model.h5')
print("✅ 모델 저장 완료!")

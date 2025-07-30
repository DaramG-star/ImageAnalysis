import tensorflow as tf

# 1. Keras 모델 로드
model = tf.keras.models.load_model("stop_hands/gesture_model6.h5")

# 2. TFLite Converter 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. (선택) 최적화 적용 - 용량 줄이고 속도 향상
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. 변환 수행
tflite_model = converter.convert()

# 5. 저장
with open("stop_hands/gesture_model6.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ gesture_model5.tflite 저장 완료!")

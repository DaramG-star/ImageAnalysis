import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import albumentations

print("✅ numpy:", np.__version__)
print("✅ cv2:", cv2.__version__)
print("✅ tf:", tf.__version__)
print("✅ mediapipe:", mp.__version__)
print("✅ albumentations:", albumentations.__version__)

# CascadeClassifier 동작 확인
print("✅ CascadeClassifier exists:", hasattr(cv2, 'CascadeClassifier'))

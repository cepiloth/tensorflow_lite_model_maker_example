import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 학습된 모델의 경로
export_dir = "model/my_saved_model"

# 학습된 모델을 가져온다
new_model = tf.keras.models.load_model(export_dir)

# 모델 구조를 확인합니다
new_model.summary()

# TFLiteConverter 로 저장된 모델을 읽어온다
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 양자화 타입을 uint8로 설정한다
converter.target_spec.supported_types = [tf.uint8]

# 텐서플로우 모델을 TFLite 모델로 변환한다
tflite_quant_uint8_model = converter.convert()

# TFLite 모델을 저장한다
open("converted_quant_uint8_model.tflite", "wb").write(tflite_quant_uint8_model)
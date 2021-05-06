import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 모델 경로
export_dir = "model/my_saved_model"

new_model = tf.keras.models.load_model(export_dir)

# 모델 구조를 확인합니다
new_model.summary()

# 기본 옵션으로 저장
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_float16_model = converter.convert()
open("converted_quant_float16_model.tflite", "wb").write(tflite_quant_float16_model)
import tensorflow as tf
import numpy as np
from PIL import Image
import io
tfrecord_file = '/Volumes/Extreme Pro/dataset/bridge_orig/1.0.0/bridge_dataset-train.tfrecord-00007-of-01024'

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
    key='steps/observation/image_0'
    value = example.features.feature[key]
    if value.HasField('bytes_list'):
        bytes_value = value.bytes_list.value[0]  # 假设只提取第一个值
        print(bytes_value)  # 这是二进制数据
        # 解码图像数据（假设它是一个JPEG图像）
        image = tf.io.decode_jpeg(bytes_value)

        # 将 Tensor 转换为 NumPy 数组
        image_np = image.numpy()

        # 或者使用 PIL 打开图像
        image_pil = Image.open(io.BytesIO(bytes_value))
        image_pil.show()
    a=1

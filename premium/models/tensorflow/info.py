import tensorflow as tf


def build_info():
    build = tf.sysconfig.get_build_info()
    print('CUDA version', build['cuda_version'])
    print('cuDNN version', build['cudnn_version'])

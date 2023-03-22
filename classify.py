import tensorflow as tf

model = tf.saved_model.load('sanskritclassifiermodel')

def predict(img):
    
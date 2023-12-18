import numpy as np
import matplotlib.pyplot as plt
import logging, os                             
logging.disable(logging.WARNING)                # Disables the warnings from tensorflow           
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Create noisy data
x_data = np.linspace(-10, 10, num=1000)
y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=1000)
print('Data created successfully')


# Validation data
x_vali = np.linspace(-10, 10, num=100)
y_vali = 0.1*x_vali*np.cos(x_vali) + 0.1*np.random.normal(size=100)

# build the tf.keras model using the keras model subclassing API
class MyModel(Model):
  def __init__(self):
    super().__init__()
    self.d1 = Dense(1, activation='linear')
    self.d2 = Dense(32, activation='relu')
    self.d3 = Dense(32, activation='relu')
    self.d4 = Dense(1, activation='linear')

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)

# Create an instance of the model
model = MyModel()

# Choose an optimizer and loss function for training
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()


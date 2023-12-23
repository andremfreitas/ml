import logging, os                             
logging.disable(logging.WARNING)                      
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import tensorflow as tf

x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as t:
  x_sq = x * x
  with t.stop_recording():
    y_sq = y * y
  z = x_sq + y_sq

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])

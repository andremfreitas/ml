import logging, os                             
logging.disable(logging.WARNING)                      
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import tensorflow as tf

'''
In contrast to the global tape controls above, the tf.stop_gradient function 
is much more precise. It can be used to stop gradients from flowing along a 
particular path, without needing access to the tape itself:
'''

x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as t:
  y_sq = y**2
  z = x**2 + tf.stop_gradient(y_sq)

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])

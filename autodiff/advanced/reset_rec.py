import logging, os                             
logging.disable(logging.WARNING)                      
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import tensorflow as tf

x = tf.Variable(2.0)
y = tf.Variable(3.0)
reset = True

with tf.GradientTape() as t:
  y_sq = y * y
  if reset:
    # Throw out all the tape recorded so far.
    t.reset()
  z = x * x + y_sq

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])

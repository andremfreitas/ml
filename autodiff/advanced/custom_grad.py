import logging, os                             
logging.disable(logging.WARNING)                      
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import tensorflow as tf

# Establish an identity operation, but clip during the gradient pass.
@tf.custom_gradient
def clip_gradients(y):
  def backward(dy):
    return tf.clip_by_norm(dy, 0.5)
  return y, backward

v = tf.Variable(2.0)
with tf.GradientTape() as t:
  output = clip_gradients(v * v)
print(t.gradient(output, v))  # calls "backward", which clips 4 to 2

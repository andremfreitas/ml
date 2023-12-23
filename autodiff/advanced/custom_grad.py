import logging, os                             
logging.disable(logging.WARNING)                      
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import tensorflow as tf

'''
The code defines a custom gradient function that performs gradient clipping
and then applies it to the square of a TensorFlow variable. The gradient of
the output with respect to the variable is computed, and the custom backward
function clips the gradient during this computation.
'''


# Establish an identity operation, but clip during the gradient pass.
@tf.custom_gradient                       #  decorator in TensorFlow that allows you to define a custom gradient for a function
def clip_gradients(y):                    #  This function takes a tensor y as input and returns y along with a custom backward function.
  def backward(dy):                       #  This function takes the gradient dy with respect to the output y and applies gradient clipping
    return tf.clip_by_norm(dy, 0.5)       #  using tf.clip_by_norm(dy, 0.5). It clips the gradient to have a maximum L2 norm of 0.5.
  return y, backward

v = tf.Variable(2.0)                      # This creates a TensorFlow variable v with an initial value of 2.0.
with tf.GradientTape() as t:              # sets up a gradient tape to record operations for automatic differentiation.
  output = clip_gradients(v * v)          #  It applies the clip_gradients function to the square of the variable v.
print(t.gradient(output, v))              # This computes the gradient of the output with respect to the variable v using the recorded
                                          # operations in the gradient tape. The custom backward function defined earlier is called during this
                                          #  gradient computation, which clips the gradient.

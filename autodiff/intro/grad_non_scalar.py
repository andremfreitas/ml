import logging, os                             
logging.disable(logging.WARNING)                      
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import tensorflow as tf
import matplotlib.pyplot as plt

'''
Gradients of non-scalar targets
'''

# A gradient is fundamentally an operation on a scalar

x = tf.Variable(2.0)
with tf.GradientTape(persistent=True) as tape:
  y0 = x**2
  y1 = 1 / x

print(tape.gradient(y0, x).numpy())
print(tape.gradient(y1, x).numpy())

# Thus, if you ask for the gradient of multiple targets, the result for each source is:
    # The gradient of the sum of the targets, or equivalently
    # The sum of the gradients of each target.

print(tape.gradient({'y0': y0, 'y1': y1}, x).numpy())


# Similarly, if the target(s) are not scalar the gradient of the sum is calculated:

x = tf.Variable(2.)

with tf.GradientTape() as tape:
  y = x * [3., 4.]

print(tape.gradient(y, x).numpy())


'''
This makes it simple to take the gradient of the sum of a collection of losses, or the gradient
 of the sum of an element-wise loss calculation.

If you need a separate gradient for each item, refer to Jacobians.

In some cases you can skip the Jacobian. For an element-wise calculation, the gradient of the sum
 gives the derivative of each element with respect to its input-element, since each element is independent:
'''

x = tf.linspace(-10.0, 10.0, 200+1)

with tf.GradientTape() as tape:
  tape.watch(x)
  y = tf.nn.sigmoid(x)

dy_dx = tape.gradient(y, x)

plt.plot(x, y, label='y')
plt.plot(x, dy_dx, label='dy/dx')
plt.legend()
plt.xlabel('x')
plt.show()
import logging, os                             
logging.disable(logging.WARNING)                      
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import tensorflow as tf

'''
CONTROL FLOW

Because a gradient tape records operations as they are executed, Python control flow is naturally handled (for example, if and while statements).

Here a different variable is used on each branch of an if. The gradient only connects to the variable that was used:

'''

x = tf.constant(1.0)

v0 = tf.Variable(2.0)
v1 = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
  tape.watch(x)
  if x > 0.0:
    result = v0
  else:
    result = v1**2 

dv0, dv1 = tape.gradient(result, [v0, v1])

print(dv0)
print(dv1)

'''
Just remember that the control statements themselves are not differentiable,
 so they are invisible to gradient-based optimizers.

Depending on the value of x in the above example, the tape either records result = v0 or result = v1**2. 
The gradient with respect to x is always None.
'''

dx = tape.gradient(result, x)

print(dx)
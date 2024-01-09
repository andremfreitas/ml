import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# We will start with a linear regression
# Our data is sampled according to:
#       y = w_{true}x + b_{true} + epsilon

xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
ys = xs * 3 - 1 + noise

#plt.scatter(xs,ys)
#plt.show()

# Therefore our model is:
#     y^ (x;theta) = wx + b

# We will use a single array theta to house both parameters:
#     theta = [w,b]

def model(theta, x):
 """Computes wx + b on a batch of input x"""
 w,b = theta
 return w * x + b

# The loss function is: J(x,y;theta) = (y^ - y)^2

def loss_fn(theta, x, y):
 prediction = model(theta, x)
 return jnp.mean((prediction-y)**2)

def update(theta, x, y, lr=0.1):
 return theta - lr * jax.grad(loss_fn)(theta, x, y)

update_jit = jax.jit(update)

theta = jnp.array([1.,1.])

for _ in range(1000):
 theta = update_jit(theta, xs, ys)

#plt.scatter(xs, ys)
#plt.plot(xs, model(theta, xs))
#plt.show()

w,b = theta
print(f"w: {w:<.2f}, b: {b:<.2f}")

import jax
import jax.numpy as jnp

def sum_of_squares(x):
 return jnp.sum(x**2)

sum_of_squares_dx = jax.grad(sum_of_squares)

x = jnp.asarray([1.0,2.0,3.0,4.0])

print(sum_of_squares(x))
print(sum_of_squares_dx(x))

def sum_squared_error(x,y):
 return jnp.sum((x-y)**2)

sum_squared_error_dx = jax.grad(sum_squared_error)

y = jnp.asarray([1.1, 2.1, 3.1, 4.1])

print(sum_squared_error_dx(x,y))

# find partial wrt both x and y
print(jax.grad(sum_squared_error, argnums = (0,1))(x,y))

# does this mean that when doing ML we need to define long functions to be able to
# obtain the gradient of the loss function wrt the different parameters ?  NO!
# JAX comes equipped with machinery for bundling arrays together ind ata structures
# called 'pytrees'.

# So most often, the use of jax.grad looks something like this:

'''
def loss_fn(params, data):
 ...

grads = jax.grad(loss_fn)(params, data_batch)
'''

# where 'params' is, for examplem a nested dict of arrays, and the returned 'grads'
# are alson a nested dict of arrays with the same structure.

## Value and Grad

# Often we need to find both the value and the grad of the function (think about the loss
# function, where we want the gradient and its value to know how training is going)

print(jax.value_and_grad(sum_squared_error)(x, y))

# which returns a tuple of (value, grad)


## Auxiliary data

# In addition to wanting the value of the function itself, sometimes, we also want
# to report some intermediate results. But if we try to do that with jax.grad, we run into trouble:

def squared_error_with_aux(x, y):
 return sum_squared_error(x, y), x-y

# jax.grad(squared_error_with_aux)(x,y)
# ^ this results in error

# instead we should do:

print(jax.grad(squared_error_with_aux, has_aux=True)(x,y))


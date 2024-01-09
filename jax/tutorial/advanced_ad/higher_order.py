import jax 
import jax.numpy as jnp

# Higher order derivatives

f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)

print(dfdx(1.))
print(d2fdx(1.))
print(d3fdx(1.))
print(d4fdx(1.))

'''
Hessian
---------

For the multivariate case, we have that the second order derivative of a function
is represented by its Hessian matrix.

The Hessian of a real-valued function of several variables, 
can be identified with the Jacobian of its gradient. JAX provides
two transformations for computing the Jacobian of a function, jax.jacfwd
and jax.jacrev, corresponding to forward- and reverse-mode autodiff.
They give the same answer, but one can be more efficient than the other
in different circumstances

I guess that if we have a jacobian with many columns (compared to rows),
then doing foward mode is benefitial, whereas if our jacobian has more 
rows than columns then reverse ad is benefitial.

'''


def hessian(f):
  return jax.jacfwd(jax.grad(f))

# Let's double check this implementation on the dot product f: x -> x x

def f(x):
 return jnp.dot(x,x)

print(hessian(f)(jnp.array([1., 2., 3.])))

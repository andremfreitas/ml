'''
Stopping Gradients
------------------

Auto-diff enables automatic computation of the gradient of a function
with respect to its inputs. Sometimes, however, we might want some
additional control: for instance, we might want to avoid back-propagating
gradients through some subset of the computational graph. 
'''

import jax
import jax.numpy as jnp

# Value function and initial parameters
value_fn = lambda theta, state: jnp.dot(theta, state)
theta = jnp.array([0.1, -0.1, 0.])

# An example transition.
s_tm1 = jnp.array([1., 2., -1.])
r_t = jnp.array(1.)
s_t = jnp.array([2., 1., 0.])

def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return -0.5 * ((jax.lax.stop_gradient(target) - v_tm1) ** 2)

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

'''
'jax.lax.stop_gradient' may also be useful in other settings,
for instance if you want the gradient from some loss to only
affect a subset of the parameters of the neural network (because,
for instance, the other parameters are trained using a different loss).
'''

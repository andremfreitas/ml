'''
Pseudo RNG in JAX is different than in NumPY
'''


from jax import random

key = random.PRNGKey(42)

print(key)


'''
‘Random key’ is essentially just another word for ‘random seed’.
However, instead of setting it once as in NumPy, any call of a random
function in JAX requires a key to be specified. Random functions consume
the key, but do not modify it. Feeding the same key to a random function
will always result in the same sample being generated: 
'''

print(random.normal(key))
print(random.normal(key))

# The rule of thumb is: never reuse keys (unless you want identical outputs).

'''
In order to generate different and independent samples, you must split()
the key yourself whenever you want to call a random function: 
'''

new_key, subkey = random.split(key)

'''
random.split()
--------------

split() is a deterministic function that converts one key into several
independent (in the pseudorandomness sense) keys. We keep one of the
outputs as the new_key, and can safely use the unique extra key (called
subkey) as input into a random function, and then discard it forever.

If you wanted to get another sample from the normal distribution, you
would split key again, and so on. The crucial point is that you never
use the same PRNGKey twice. Since split() takes a key as its argument,
we must throw away that old key when we split it.

It doesn’t matter which part of the output of split(key) we call key,
and which we call subkey. They are all pseudorandom numbers with equal
status. The reason we use the key/subkey convention is to keep track of
how they’re consumed down the road. Subkeys are destined for immediate
consumption by random functions, while the key is retained to generate
more randomness later.
'''

# To discard the old key automatically, one could do:
key, subkey = random.split(key)

# It is worth noting that split() can create as many keys as needed:

key, *twenty_subkeys = random.split(key, num=21)

'''
Another difference between NumPy’s and JAX’s random modules relates
to the sequential equivalence guarantee mentioned above.

As in NumPy, JAX’s random module also allows sampling of vectors of
numbers. However, JAX does not provide a sequential equivalence guarantee,
because doing so would interfere with the vectorization on SIMD hardware. 
'''
import numpy as np

key = random.PRNGKey(42)
subkeys = random.split(key, 3)
sequence = np.stack([random.normal(subkey) for subkey in subkeys])
print("individually:", sequence)

key = random.PRNGKey(42)
print("all at once: ", random.normal(key, shape=(3,)))

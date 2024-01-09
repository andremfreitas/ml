import jax
import jax.numpy as jnp
from typing import Tuple

class Counter:
 """A simple counter."""

 def __init__(self) -> None:
       self.n = 0

 def count(self) -> int:
  """Increments the counter and returns the new value"""
  self.n += 1
  return self.n

 def reset(self) -> None:
  """Resets the counter to zero"""
  self.n = 0


counter = Counter()

for _ in range(3):
 print(counter.count())
  
# Now lets try jitting this

counter.reset()
fast_count = jax.jit(counter.count)

for _ in range(3):
 print(fast_count())

'''
Solution: explicit state
------------------------ 
'''

CounterState = int

class CounterV2:
 """Counter that can be jitted"""
 def count(self, n: CounterState) -> Tuple[int, CounterState]:
  # You could just return n+1, but here we separate its role as 
  # the output and as the counter state for didactic purposes.
  return n+1, n+1

 def reset(self) -> CounterState:
  return 0

counter = CounterV2()
state = counter.reset()

for _ in range(3):
 value, state = counter.count(state)
 print(value)

#jit it
state = counter.reset()
fast_count = jax.jit(counter.count)

for _ in range(3):
  value, state = fast_count(state)
  print(value)

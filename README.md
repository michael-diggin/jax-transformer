# jax-transformer
A transformer implemented naively in Jax

Jax is very functional programming based, so create Modules (without copying Flax)
that achieve 'storing' parameters and variables in a nice way that work with Jax
and make training easy to follow
Models are a stack of Modules (eg layers) which, 
given a set of params/variables and input, give an ouput. 
Then construct a loss, collect grads and update the params/variables. 
Repeat until done. 


Other bits to do:
- Implement Dropout (Done)
- Implement Adam optimizer (Done)
- Implement gradient clipping (Done)
- Implement LR warmup and decay (Done)
- Define loss function (Done)
- Implement data prep object (Done)
- Implement training loops with validation (Done)
- Implement regularisation with weight decay
- Implement KV caching for fast decoding
- Saving/checkpointing model weights and optimizer state
- Profile and see what can be improved/made faster and see what gets recompiled unnecessarily

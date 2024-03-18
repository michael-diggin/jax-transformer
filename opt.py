import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing import NamedTuple
import modules as mod


class OptState(NamedTuple):
    train_iter: int = 0
    momentums: list[jax.Array] = []
    velocities: list[jax.Array] = []

@register_pytree_node_class
class VanillaSGD:
    def __init__(self, lr_func) -> None:
        self.lr_func = lr_func

    def tree_flatten(self):
        return [], [self.lr_func]
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    @jax.jit
    def apply_grads(self, params, grads, state: OptState):
        step = state.train_iter + 1
        lr = self.lr_func(step)
        new_params = jax.tree_map(
            lambda param, g: param - lr*g, params, grads
        )
        state = OptState(train_iter=step)
        return new_params, state
    
    def init_state(self, dummy_params) -> OptState:
        return OptState(train_iter=0)

# TODO: this only needs to be a pytree for tracing reasons, see if there's a better way to achieve this
@register_pytree_node_class
class Adam(object):
    def __init__(self, lr_func, beta1: jax.Array = 0.9, beta2: jax.Array = 0.995, grad_clip: jax.Array = jnp.inf) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.clip = grad_clip
        self.lr_func = lr_func

    @jax.jit
    def apply_grads(self, params: mod.BaseModule, grads: mod.BaseModule, state: OptState) -> tuple[mod.BaseModule, OptState]:
        step = state.train_iter + 1
        lr = self.lr_func(step)
        updates, state = self._update_opt_state(grads, state)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p - lr*u, params, updates
        )
        return new_params, state

    def _update_opt_state(self, grads, state: OptState) -> OptState:
        # t <- t +1
        # m <- (beta1*m + (1-beta1)*grad) / (1-beta1^t)
        # v <- (beta2*v + (1-beta2)*grad^2) / (1-beta2^t)
        t = state.train_iter + 1
        def update_momentum(m, g):
            g = jnp.clip(g, -self.clip, self.clip)
            m = self.beta1*m + (1-self.beta1)*g
            return m / (1 - jnp.power(self.beta1, t))
        
        def update_velocity(v, g):
            g = jnp.clip(g, -self.clip, self.clip)
            v = self.beta2*v + (1-self.beta2)*(jnp.power(g, 2))
            return v / (1 - jnp.power(self.beta2, t))

        grad_leaves, grad_tree_def = jax.tree_util.tree_flatten(grads)

        # TODO: it would be nice if both of these could be done together
        new_momentum = jax.tree_util.tree_map(
            update_momentum, state.momentums, grad_leaves
        )
        new_velocity = jax.tree_util.tree_map(
            update_velocity, state.velocities, grad_leaves
        )
        state = OptState(t, new_momentum, new_velocity)

        # gradient update is
        # p <- p - lr*m/(sqrt(v) + eps)
        # compute this on the momentum/velocity lists
        # and create as the grad tree type (which is the 'Model' type)
        grad_updates = jax.tree_util.tree_map(
            lambda m, v: m/(jnp.sqrt(v)+10e-8), new_momentum, new_velocity
        )
        grad_updates = jax.tree_util.tree_unflatten(grad_tree_def, grad_updates)
        return grad_updates, state
    
    def init_state(self, dummy_params: jax.Array) -> OptState:
        param_leaves, _ = jax.tree_util.tree_flatten(dummy_params)
        mo = [jnp.zeros_like(l) for l in param_leaves]
        ve = [jnp.zeros_like(l) for l in param_leaves]
        return OptState(train_iter=0, momentums=mo, velocities=ve)
    
    def tree_flatten(self):
        return [], [self.lr_func, self.beta1, self.beta2, self.clip]
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


def standard_lr(base_lr: int):
    return lambda x: base_lr

def cosine_lr_decay(base_lr: int, min_lr: int, warm_up: int, decay_iters: int):
    '''
    Returns a function(int) -> LR, which returns the learning rate given
    '''
    # Create function to return lr based on iteration count
    def get_lr(train_iter):

        wu_lr = base_lr * train_iter / warm_up
        lr_factor = 0.5 * (1 + jnp.cos(jnp.pi * (train_iter - warm_up) / (decay_iters - warm_up)))
        decay_lr = min_lr + lr_factor * (base_lr - min_lr)

        lr = jnp.where(train_iter <= warm_up, wu_lr, decay_lr)
        lr = jnp.where(train_iter >= decay_iters, min_lr, lr)
        return lr
    return get_lr

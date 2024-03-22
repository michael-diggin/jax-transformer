import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing import NamedTuple
import modules as mod


class OptState(NamedTuple):
    train_iter: int = 0
    momentums: mod.BaseModule = None
    velocities: mod.BaseModule = None

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


@register_pytree_node_class
class Adam(object):
    def __init__(self, lr_func, beta1: jax.Array = 0.9, beta2: jax.Array = 0.99,
                 grad_clip: jax.Array = jnp.inf, weight_decay: jax.Array = 0.) -> None:
        self.lr_func = lr_func
        self.beta1 = beta1
        self.beta2 = beta2
        self.clip = grad_clip
        self.weight_decay = weight_decay

    @jax.jit
    def apply_grads(self, params: mod.BaseModule, grads: mod.BaseModule, state: OptState) -> tuple[mod.BaseModule, OptState]:
        step = state.train_iter + 1
        lr = self.lr_func(step)
        grads = self._apply_weight_decay(params, grads)
        updates, state = self._update_opt_state(grads, state)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p - lr*u, params, updates
        )
        return new_params, state

    def _apply_weight_decay(self, params, grads) -> mod.BaseModule:
        def decay(p, g):
            return jnp.where(p.ndim >= 2, g + self.weight_decay*p, g)

        grads = jax.tree_util.tree_map(
            decay, params, grads
        )
        return grads

    def _update_opt_state(self, grads, state: OptState) -> tuple[mod.BaseModule, OptState]:
        # t <- t +1
        # m <- (beta1*m + (1-beta1)*grad) / (1-beta1^t)
        # v <- (beta2*v + (1-beta2)*grad^2) / (1-beta2^t)
        t = state.train_iter + 1
        def update_momentum(m, g):
            g = jnp.clip(g, -self.clip, self.clip)
            m = self.beta1*m + (1-self.beta1)*g
            return m
        
        def update_velocity(v, g):
            g = jnp.clip(g, -self.clip, self.clip)
            v = self.beta2*v + (1-self.beta2)*(jnp.power(g, 2))
            return v

        new_momentum = jax.tree_util.tree_map(
            update_momentum, state.momentums, grads
        )
        new_velocity = jax.tree_util.tree_map(
            update_velocity, state.velocities, grads
        )
        state = OptState(t, new_momentum, new_velocity)

        # gradient update is
        # p <- p - lr*m/(sqrt(v) + eps)
        # scalar here is the bias correction term that -> 1 as t -> inf
        # it's done here as since beta2 is close to 1, 1/(1-beta2**t) can be large for small t and
        # lead to incorrect updates
        scalar = jnp.sqrt(1 - self.beta2**t) / (1 - self.beta1**t)
        grad_updates = jax.tree_util.tree_map(
            lambda m, v: scalar * (m/(jnp.sqrt(v)+10e-8)), new_momentum, new_velocity
        )
        return grad_updates, state
    
    def init_state(self, dummy_params: jax.Array) -> OptState:
        mo = jax.tree_util.tree_map(
            jnp.zeros_like, dummy_params
        )
        ve = jax.tree_util.tree_map(
            jnp.zeros_like, dummy_params
        )
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

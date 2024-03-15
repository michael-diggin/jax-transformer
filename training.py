# training.py contains the training code for JaxFormer
import jax
import jax.numpy as jnp
from functools import partial

from data import TrainingDataset
from model import JaxFormer
from opt import Adam, OptState, consine_lr_decay
from loss import cross_entropy_loss


'''
Steps:
1. Load and prepare data
2. Init model and optimizer state
3. Run update loop
4. Validate at a set frequency
5. Checkpoint at a set frequency
'''

DATA_FILE = ".training/data.txt"

MAX_SEQ_LEN = 128
MODEL_DIM = 32
NUM_TRANSFORMER_BLOCKS = 2
HEADS = 8
HIDDEN_DIM = 2*MODEL_DIM
DROPOUT_PROB = 0.5

BASE_LR = 5e-4
WARMUP_ITERS = 100
MAX_ITERS = 30000

BATCH_SIZE = 8

@partial(jax.jit, static_argnames=["optim"])
def update_step(model: JaxFormer, optim: Adam, opt_state: OptState, X: jax.Array, Y: jax.Array, rng: jax.Array) -> tuple[jax.Array, JaxFormer, OptState]:

    def _loss(model, X):
        pred = model(X, mask=True, train=True, rng=rng)
        return cross_entropy_loss(pred, Y)
    
    loss, grads = jax.value_and_grad(_loss)(model, X)
    model, opt_state = optim.apply_grads(model, grads, opt_state)
    return loss, model, opt_state
    

def validation_loss(model, iterations) -> jax.Array:
    pass



if __name__ == "__main__":

    ds = TrainingDataset(DATA_FILE)
    vocab_size = ds.vocab_size()

    rng = jax.random.key(101)
    rng, model_key = jax.random.split(rng)

    jax_model = JaxFormer.init(model_key, MAX_SEQ_LEN, vocab_size, MODEL_DIM, 
                               NUM_TRANSFORMER_BLOCKS, HEADS, HIDDEN_DIM, DROPOUT_PROB)
    
    dummy_model_params = jax.tree_util.tree_leaves(jax_model)

    lr_decay = consine_lr_decay(BASE_LR, WARMUP_ITERS, MAX_ITERS)
    optim = Adam(lr_decay, grad_clip=1.0)
    opt_state = optim.init_state(dummy_model_params)

    step = 0
    while step < MAX_ITERS:
        rng, ds_key, update_key = jax.random.split(rng, 3)
        X, Y = ds.get_batch(ds_key, BATCH_SIZE, MAX_SEQ_LEN)

        loss, jax_model, opt_state = update_step(jax_model, optim, opt_state, X, Y, update_key)
        print(loss)
        step += 1
    

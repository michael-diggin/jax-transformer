# training.py contains the training code for JaxFormer
import jax
import jax.numpy as jnp
from functools import partial

from data import TrainingDataset
from model import JaxFormer
from opt import Adam, OptState, cosine_lr_decay
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

MAX_SEQ_LEN = 256
MODEL_DIM = 384 # 6*64
NUM_TRANSFORMER_BLOCKS = 6
HEADS = 6
HIDDEN_DIM = 4*MODEL_DIM
DROPOUT_PROB = 0.2

BASE_LR = 0.001
MIN_LR = BASE_LR / 10.
WARMUP_ITERS = 100
DECAY_ITERS = 2500
WEIGHT_DECAY = 0.001

BATCH_SIZE = 64

EVAL_ITERS = 10
EVAL_FREQ = 100
NUM_STEPS = 4000
FREQ = 10

@partial(jax.jit, static_argnames=["optim"])
def update_step(model: JaxFormer, optim: Adam, opt_state: OptState, X: jax.Array, Y: jax.Array, rng: jax.Array) -> tuple[jax.Array, JaxFormer, OptState]:

    def _loss(model, X):
        pred = model(X, train=True, rng=rng)
        return cross_entropy_loss(pred, Y)
    
    loss, grads = jax.value_and_grad(_loss)(model, X)
    model, opt_state = optim.apply_grads(model, grads, opt_state)
    return loss, model, opt_state
    
def validation_loss(ds: TrainingDataset, model: JaxFormer, iterations: int, rng: jax.Array) -> jax.Array:
    losses = jnp.zeros((iterations))
    for k in range(iterations):
        rng, ds_rng = jax.random.split(rng)
        X, Y = ds.get_batch(ds_rng, BATCH_SIZE, MAX_SEQ_LEN, "val")
        logits = model(X, train=False)
        loss = cross_entropy_loss(logits, Y)
        losses = losses.at[k].set(loss)

    return losses.mean()

def generate(model, start: str, encoder, decoder, rng: jax.Array, num: int):
    '''
    This function should take in an array of tokens of shape (1, MAX_SEQ_LEN)
    This is a mild hack because otherwise we retrace the model call function
    for every new input shape
    TODO: could use pad tokens here?
    '''
    tokens = start

    for _ in range(num):
        rng, new = jax.random.split(rng)
        tokens = tokens[:, -num:]
        preds = model(tokens, train=False)
        # preds (1, seq_len, vocab_size)
        preds = preds[:, -1, :]
        choice = jax.random.categorical(new, preds, axis=-1) # this applies softmax
        tokens = jnp.append(tokens, jnp.expand_dims(choice, axis=0), axis=1)

    return decoder(tokens[0])





if __name__ == "__main__":

    ds = TrainingDataset(DATA_FILE)
    vocab_size = ds.vocab_size()

    rng = jax.random.key(111)
    rng, model_key = jax.random.split(rng)

    jax_model = JaxFormer.init(model_key, MAX_SEQ_LEN, vocab_size, MODEL_DIM, 
                               NUM_TRANSFORMER_BLOCKS, HEADS, HIDDEN_DIM, DROPOUT_PROB)

    lr_decay = cosine_lr_decay(BASE_LR, MIN_LR, WARMUP_ITERS, DECAY_ITERS)
    optim = Adam(lr_decay, grad_clip=1.0, weight_decay=WEIGHT_DECAY)
    opt_state = optim.init_state(jax_model)

    freq_losses = jnp.zeros((FREQ,))
    step_losses = jnp.array([])
    val_losses = jnp.array([])

    step = 0
    while step < NUM_STEPS:
        rng, ds_key, update_key = jax.random.split(rng, 3)
        X, Y = ds.get_batch(ds_key, BATCH_SIZE, MAX_SEQ_LEN)

        loss, jax_model, opt_state = update_step(jax_model, optim, opt_state, X, Y, update_key)
        step_losses = jnp.append(step_losses, loss)


        if step > 0 and step%FREQ == 0:
            print(f"Step: {step}. Loss over last {FREQ} steps: {jnp.mean(freq_losses):.4f}")
        
        freq_losses = freq_losses.at[step%FREQ].set(loss)

        if step > 0 and step%EVAL_FREQ == 0:
            rng, val_key = jax.random.split(rng)
            val_loss = validation_loss(ds, jax_model, EVAL_ITERS, val_key)
            val_losses = jnp.append(val_losses, val_loss)
            print(f"Validation Loss: {val_loss:.4f}")

        step += 1


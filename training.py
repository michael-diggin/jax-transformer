# training.py contains the training code for JaxFormer
import jax
import jax.numpy as jnp
from functools import partial

from data import TrainingDataset
from model import JaxFormer
from opt import Adam, VanillaSGD, OptState, cosine_lr_decay, standard_lr
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

BASE_LR = 0.01
MIN_LR = BASE_LR / 100.
WARMUP_ITERS = 300
DECAY_ITERS = 3000

BATCH_SIZE = 32

EVAL_ITERS = 10
EVAL_FREQ = 100
NUM_STEPS = 4000
FREQ = 10

@partial(jax.jit, static_argnames=["optim"])
def update_step(model: JaxFormer, optim: Adam, opt_state: OptState, X: jax.Array, Y: jax.Array, rng: jax.Array) -> tuple[jax.Array, JaxFormer, OptState]:

    def _loss(model, X):
        pred = model(X, mask=True, train=True, rng=rng)
        return cross_entropy_loss(pred, Y)
    
    loss, grads = jax.value_and_grad(_loss)(model, X)
    model, opt_state = optim.apply_grads(model, grads, opt_state)
    return loss, model, opt_state
    
def validation_loss(ds: TrainingDataset, model: JaxFormer, iterations: int, rng: jax.Array) -> jax.Array:
    losses = jnp.zeros((iterations))
    for k in range(iterations):
        rng, ds_rng = jax.random.split(rng)
        X, Y = ds.get_batch(ds_rng, BATCH_SIZE, MAX_SEQ_LEN, "val")
        logits = model(X, mask=False, train=False, rng=None)
        loss = cross_entropy_loss(logits, Y)
        losses = losses.at[k].set(loss)

    return losses.mean()

def generate(model, start: str, encoder, decoder, rng: jax.Array):
    output = encoder(start)

    while len(output) < MAX_SEQ_LEN:
        rng, new = jax.random.split(rng)
        batch = jnp.expand_dims(jnp.array(output), axis=0)
        preds = model(batch, mask=False)
        # preds (1, seq_len, vocab_size)
        preds = preds[:, [-1], :]
        choice = jax.random.categorical(new, preds, axis=-1) # this applies softmax
        tok = choice[0][0]
        output.append(tok)

    return decoder(output)





if __name__ == "__main__":

    ds = TrainingDataset(DATA_FILE)
    vocab_size = ds.vocab_size()

    rng = jax.random.key(111)
    rng, model_key = jax.random.split(rng)

    jax_model = JaxFormer.init(model_key, MAX_SEQ_LEN, vocab_size, MODEL_DIM, 
                               NUM_TRANSFORMER_BLOCKS, HEADS, HIDDEN_DIM, DROPOUT_PROB)
    
    dummy_model_params = jax.tree_util.tree_leaves(jax_model)

    lr_decay = cosine_lr_decay(BASE_LR, MIN_LR, WARMUP_ITERS, DECAY_ITERS)
    #optim = Adam(lr_decay)
    optim = VanillaSGD(lr_decay)
    opt_state = optim.init_state(dummy_model_params)

    losses = jnp.zeros((FREQ,))
    

    step = 0
    while step < NUM_STEPS:
        rng, ds_key, update_key = jax.random.split(rng, 3)
        X, Y = ds.get_batch(ds_key, BATCH_SIZE, MAX_SEQ_LEN)

        loss, jax_model, opt_state = update_step(jax_model, optim, opt_state, X, Y, update_key)


        if step > 0 and step%FREQ == 0:
            print(f"Step: {step}. Loss over last {FREQ} steps: {jnp.mean(losses):.4f}")
        
        losses = losses.at[step%FREQ].set(loss)

        if step > 0 and step%EVAL_FREQ == 0:
            rng, val_key = jax.random.split(rng)
            val_loss = validation_loss(ds, jax_model, EVAL_ITERS, val_key)
            print(f"Validation Loss: {val_loss:.4f}")

        step += 1

    enc = ds.encoder()
    dec = ds.decoder()
    start = '\n'
    
    output = generate(jax_model, start, enc, dec, rng)
    print(output)

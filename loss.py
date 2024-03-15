import jax
import jax.numpy as jnp

'''
categorical cross entropy
should ignore pad tokens

Input is (batch, seq_len, vocab_size)
Targets will be (batch, seq_len), after encoding like the training data input
can make the targets into (batch, seq_len, vocab_size) by OHE
input needs softmax across the last dim
'''

@jax.jit
def cross_entropy_loss(logits: jax.Array, targets: jax.Array, mask: jax.Array = None) -> jax.Array:
    '''
    computes the cross entropy loss between the logits and the targets
    mask is used to ignore padded tokens in the batch
    mask should be the same shape as targets (batch, batch_seq_len)
    '''
    vocab_size = logits.shape[-1]
    targets = jax.nn.one_hot(targets, vocab_size)
    logits = jax.nn.log_softmax(logits, axis=-1, where=mask, initial=0.0)
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1, vocab_size)
    return -jnp.mean(jnp.sum(logits*targets, axis=-1))


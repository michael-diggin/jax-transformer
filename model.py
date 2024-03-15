from functools import partial
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import modules as mod

@register_pytree_node_class
class JaxFormer(mod.BaseModule):
    def __init__(self, embedding: mod.Embedding, transformers: list[mod.Transformer], lm_head: mod.Dense, *args):
        self.embedding = embedding
        self.transformers = transformers
        self.lm_head = lm_head

    def params(self):
        return [self.embedding, self.transformers, self.lm_head]

    @partial(jax.jit, static_argnames=["mask", "train"])
    def __call__(self, x: jax.Array, mask: bool, train: bool = False, rng: jax.Array = None) -> jax.Array:
        # x is (batch, seq_len)
        x = self.embedding(x) # (batch, seq_len, d_model), this includes the pos encodings
        block_rng = None
        for tformer in self.transformers:
            if train:
                rng, block_rng = jax.random.split(rng)
            x = tformer(x, mask, train, block_rng) # (batch, seq_len, d_model)
        x = self.lm_head(x) # (batch, seq_len, vocab_size)
        return x

    @classmethod
    def init(cls, rng: jax.Array, max_seq_len: int, vocab_size: int, d_model: int,
            num_blocks: int, heads: int, hidden_dim: int, dropout_prob: int):
        rng, e_key, dense_key = jax.random.split(rng, 3)
        embed = mod.Embedding.init(e_key, max_seq_len, d_model, vocab_size)
        d_blocks = []
        for _ in range(num_blocks):
            rng, block_key = jax.random.split(rng)
            d_blocks.append(mod.Transformer.init(block_key, d_model, heads, hidden_dim, dropout_prob))
        lm_head = mod.Dense.init(dense_key, d_model, vocab_size)
        return cls(embed, d_blocks, lm_head)
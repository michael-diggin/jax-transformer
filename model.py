from functools import partial
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import modules as mod

@register_pytree_node_class
class JaxFormer(mod.BaseModule):
    def __init__(self, embedding: mod.Embedding, drop: mod.Dropout, transformers: list[mod.Transformer], norm: mod.LayerNorm, lm_head: mod.Dense, *args):
        self.embedding = embedding
        self.drop = drop
        self.transformers = transformers
        self.norm = norm
        self.lm_head = lm_head

    def params(self):
        return [self.embedding, self.drop, self.transformers, self.norm, self.lm_head]

    @partial(jax.jit, static_argnames=["train"])
    def __call__(self, x: jax.Array, train: bool = False, rng: jax.Array = None) -> jax.Array:
        # x is (batch, seq_len)
        x = self.embedding(x) # (batch, seq_len, d_model), this includes the pos encodings
        if train:
            rng, drop_key = jax.random.split(rng)
            x = self.drop(x, train, drop_key)
        block_rng = None
        for tformer in self.transformers:
            if train:
                rng, block_rng = jax.random.split(rng)
            x = tformer(x, train, block_rng) # (batch, seq_len, d_model)
        x = self.norm(x)
        x = self.lm_head(x) # (batch, seq_len, vocab_size)
        return x

    @classmethod
    def init(cls, rng: jax.Array, max_seq_len: int, vocab_size: int, d_model: int,
            num_blocks: int, heads: int, hidden_dim: int, dropout_prob: int, kernel_std: float = 0.02, with_bias: bool = True):
        rng, e_key, dense_key = jax.random.split(rng, 3)
        embed = mod.Embedding.init(e_key, max_seq_len, d_model, vocab_size, kernel_std)
        drop = mod.Dropout.init(dropout_prob)
        d_blocks = []
        for _ in range(num_blocks):
            rng, block_key = jax.random.split(rng)
            d_blocks.append(mod.Transformer.init(block_key, d_model, heads, hidden_dim, max_seq_len, dropout_prob, kernel_std, with_bias))
        norm = mod.LayerNorm.init(d_model, with_bias=with_bias)
        lm_head = mod.Dense.init(dense_key, d_model, vocab_size, kernel_std, with_bias)
        embed.embedding = jnp.transpose(lm_head.weights) # weight tying
        return cls(embed, drop, d_blocks, norm, lm_head)
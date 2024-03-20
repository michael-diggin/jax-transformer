import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from functools import partial


'''
Need a better way to handle the 'params' (model trainable variables)
Currently ~okay but quite fragile (eg ordering matters)
Could use a dictionary in some way
'''


@register_pytree_node_class
class BaseModule(object):
    def __init__(self, *args, **kwargs):
        pass

    def params(self):
        pass

    def aux_data(self):
        return None

    def tree_flatten(self):
        return self.params(), self.aux_data()
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(*children, aux_data)
        return obj
    
    def __call__(self, x):
        raise NotImplemented
    
    @classmethod
    def init(cls, *args, **kwards):
        raise NotImplemented
    
@register_pytree_node_class
class Embedding(BaseModule):
    def __init__(self, embedding: jax.Array, positional_encodings: jax.Array = None, *args) -> None:
        self.embedding = embedding
        self.pos_encodings = positional_encodings
    
    def params(self) -> list[jax.Array]:
        return [self.embedding]
    
    def aux_data(self) -> jax.Array:
        # positional encodings are constants rather than learnable params
        return self.pos_encodings

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def __call__(self, x) -> jax.Array:
        '''
        Take in the tokens, NOT one-hot encoded
        Project using the embeddings
        Add the positional encoding values
        '''
        # x (batch, seq_len)
        batch_seq_len = x.shape[-1]
        x = jnp.take(self.embedding, x, axis=0) # (batch, seq_len, embed_dim)
        x = x * jnp.sqrt(x.shape[-1]) # times sqrt(d_model=embed_dim)
        x = x + self.pos_encodings[:batch_seq_len, :]  # (batch, seq_len, embed_dim) with broadcasting
        return x
    
    @classmethod
    def init(cls, rng: jax.Array, max_seq_len: int, embedding_dim: int, vocab_size: int, kernel_std: float = 1.):
        embedding = jax.random.normal(rng, (vocab_size, embedding_dim)) * kernel_std
        pos_encoding = cls.create_pe(max_seq_len, embedding_dim)
        return cls(embedding, pos_encoding)
    
    @classmethod
    def create_pe(cls, max_seq_len: int, embedding_dim: int) -> jax.Array:
        pe = jnp.zeros(shape=(max_seq_len, embedding_dim))
        pos = jnp.arange(0, max_seq_len, dtype=jnp.float32)[:, None]
        exponent = jnp.arange(0, embedding_dim, 2, dtype=jnp.float32) / embedding_dim
        div_term = jnp.power(1.0/10000.0, exponent)
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))
        return pe
    
@register_pytree_node_class
class Dense(BaseModule):
    def __init__(self, weights: jax.Array, bias: jax.Array, with_bias: bool, *args) -> None:
        self.weights = weights
        self.bias =  bias
        self.with_bias = with_bias

    def params(self) -> list[jax.Array]:
        return [self.weights, self.bias]
    
    def aux_data(self) -> jax.Array:
        return self.with_bias
    
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def __call__(self, x) -> jax.Array:
        x = jnp.matmul(x, self.weights)
        return jnp.where(self.with_bias, x+self.bias, x)
    
    @classmethod
    def init(cls, rng: jax.Array, input: int, features: int, kernel_std: float = 1., with_bias: bool = True):
        weights = jax.random.normal(rng, (input, features)) * kernel_std
        bias = jnp.zeros((features,))
        return cls(weights, bias, with_bias)

@register_pytree_node_class
class LayerNorm(BaseModule):
    def __init__(self, learned_lambda: jax.Array, learned_beta: jax.Array, aux_data, *args) -> None:
        self.learned_lambda = learned_lambda
        self.learned_beta = learned_beta
        self.eps = aux_data[0]
        self.with_bias = aux_data[1]

    def params(self) -> list[jax.Array]:
        return [self.learned_lambda, self.learned_beta]
    
    def aux_data(self) -> list[jax.Array]:
        return [self.eps, self.with_bias]
    
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def __call__(self, x) -> jax.Array:
        mean = jnp.mean(x)
        var = jnp.var(x)
        out = ((x - mean) / jnp.sqrt(var + self.eps)) * self.learned_lambda
        return jnp.where(self.with_bias, out + self.learned_beta, out)

    @classmethod
    def init(cls, d_model: int, eps: int = 10e-09, with_bias: bool = True):
        # initialized with 1 and 0
        lambda_p = jnp.ones((d_model,))
        beta_p = jnp.zeros((d_model,))
        return cls(lambda_p, beta_p, [eps, with_bias])
    
@register_pytree_node_class
class Dropout(BaseModule):
    def __init__(self, rate: jax.Array) -> None:
        self.rate = rate

    def params(self):
        return []
    
    def aux_data(self):
        return self.rate

    @partial(jax.jit, static_argnames=["train"])
    def __call__(self, x: jax.Array, train: bool = False, rng: jax.Array = None) -> jax.Array:
        if self.rate == 0.0 or not train:
            return x
        if self.rate == 1.0:
            return jnp.zeros_like(x)

        kp = 1 - self.rate
        mask = jax.random.bernoulli(rng, kp, x.shape)
        return jnp.where(mask, x / kp, jnp.zeros_like(x))

    @classmethod
    def init(cls, rate: jax.Array):
        return cls(rate)

@register_pytree_node_class
class MultiHeadAttention(BaseModule):
    def __init__(self, W_q: Dense, W_k: Dense, W_v: Dense, W_o: Dense, drop: Dropout, heads: int, *args) -> None:
        self.W_q = W_q
        self.W_k = W_k
        self.W_v = W_v
        self.W_o = W_o
        self.drop = drop
        self.heads = heads

    def params(self) -> list[jax.Array]:
        return [self.W_q, self.W_k, self.W_v, self.W_o, self.drop]
    
    def aux_data(self) -> jax.Array:
        return self.heads
    
    @partial(jax.jit, static_argnames=["train"])
    def __call__(self, x: jax.Array, train: bool = False, rng: jax.Array = None) -> jax.Array:
        # x is (batch size, seq_len, d_model)
        batch, seq_len, d_model = jnp.shape(x)
        q = self.W_q(x).reshape(batch, seq_len, self.heads, -1) # (batch, seq_len, heads, d_k)
        k = self.W_k(x).reshape(batch, seq_len, self.heads, -1) # (batch, seq_len, heads, d_k)
        v = self.W_v(x).reshape(batch, seq_len, self.heads, -1) # (batch, seq_len, heads, d_k)

        q = q.transpose(0, 2, 1, 3) # (batch, heads, seq_len, d_k)
        k = k.transpose(0, 2, 1, 3) # (batch, heads, seq_len, d_k)
        v = v.transpose(0, 2, 1, 3) # (batch, heads, seq_len, d_k)
        attn = self.scaled_dot_product(q, k, v, d_model // self.heads, train, rng) # (batch, heads, seq_len, d_k)
        attn = attn.transpose(0, 2, 1, 3) # (batch, seq_len, heads, d_k)
        attn = attn.reshape(batch, seq_len, d_model)
        attn_o = self.W_o(attn) # (batch, seq_len, d_model)
        return attn_o

    def scaled_dot_product(self, q: jax.Array, k: jax.Array, v: jax.Array, d_k: jax.Array, train: bool, rng: jax.Array) -> jax.Array:
        # q, v, k have dim (batch, heads, seq_len, d_k)
        attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k) # (batch, heads, seq_len, seq_len)
        m_array = self.get_training_mask(q.shape[2])
        attn = jnp.where(m_array, attn, -9e15)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.drop(attn, train, rng)
        return jnp.matmul(attn, v) # (batch, heads, seq_len, d_k)

    def get_training_mask(self, seq_len: int) -> jax.Array:
        '''
        Returns a (1, 1, seq_len, seq_len) lower trinagular mask
        that can be used in training
        '''
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return jnp.expand_dims(mask, (0, 1))

    @classmethod
    def init(cls, rng: jax.Array, heads: int, d_model: int, drop_rate: jax.Array, kernel_std: float = 1., with_bias: bool =True):
        # create all proj matrices, have d_keys = d_vals
        assert d_model % heads == 0, "Heads must divide the model dimension evenly"
        rng, q_key, k_key, v_key, o_key = jax.random.split(rng, 5)
        W_q = Dense.init(q_key, d_model, d_model, kernel_std, with_bias)
        W_k = Dense.init(k_key, d_model, d_model, kernel_std, with_bias)
        W_v = Dense.init(v_key, d_model, d_model, kernel_std, with_bias)
        W_o = Dense.init(o_key, d_model, d_model, kernel_std, with_bias)
        drop = Dropout.init(drop_rate)
        return cls(W_q, W_k, W_v, W_o, drop, heads)
    
@register_pytree_node_class
class Transformer(BaseModule):
    def __init__(self, mha: MultiHeadAttention, norm1: LayerNorm, norm2: LayerNorm,
                dense1: Dense, dense2: Dense,
                dropout1: Dropout, dropout2: Dropout, *args) -> None:
        self.mha = mha
        self.norm1 = norm1
        self.norm2 = norm2
        self.dense1 = dense1
        self.dense2 = dense2
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    def params(self) -> jax.Array:
        return [self.mha, self.norm1, self.norm2, self.dense1, self.dense2, self.dropout1, self.dropout2]

    @partial(jax.jit, static_argnames=["train"])
    def __call__(self, x: jax.Array, train: bool = False, rng: jax.Array = None) -> jax.Array:
        '''
        compute attention with MHA
        residual add + layer norm
        pass to a dense layer with hidden_dims
        relu
        pass to second dense layer back to d_model
        add and norm
        '''
        rng1, rng2, rng3 = None, None, None
        if train:
            rng1, rng2, rng3 = jax.random.split(rng, 3)
        x = self.norm1(x)
        attn = self.mha(x, train, rng1)
        attn = self.dropout1(attn, train, rng2)
        x = self.norm2(x + attn)

        ff_out = self.dense1(x)
        ff_out = jax.nn.gelu(ff_out)
        ff_out = self.dense2(ff_out)
        ff_out = self.dropout2(ff_out, train, rng3)

        return x + ff_out

    @classmethod
    def init(cls, rng: jax.Array, d_model: int, heads: int, hidden_dim: int, dropout_prob: int, kernel_std: float = 1., with_bias: bool = True):
        rng, mha_key, dense1_key, dense2_key = jax.random.split(rng, 4)
        mha = MultiHeadAttention.init(mha_key, heads, d_model, dropout_prob, kernel_std, with_bias)
        dense1 = Dense.init(dense1_key, d_model, hidden_dim, kernel_std, with_bias)
        dense2 = Dense.init(dense2_key, hidden_dim, d_model, kernel_std, with_bias)
        ln1 = LayerNorm.init(d_model, with_bias=with_bias)
        ln2 = LayerNorm.init(d_model, with_bias=with_bias)
        drop1 = Dropout.init(dropout_prob)
        drop2 = Dropout.init(dropout_prob)
        return cls(mha, ln1, ln2, dense1, dense2, drop1, drop2)

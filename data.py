import jax
from jax import vmap, lax
import jax.numpy as jnp
from functools import partial
'''
Utility function to fetch information like
char <-> int mappings, get the vocab,
get the training/val set etc
'''

class TrainingDataset:
    def __init__(self, from_file: str, validation_split: int = 0.9) -> None:
        self.data = self._read(from_file)
        self.size = len(self.data)
        self.val_split = validation_split
        self._vocab_size = None
        self.ch_int = None
        self.int_ch = None
        self._train_data = None
        self._val_data = None

    def _read(self, filename) -> str:
        with open(filename, 'r') as f:
            data = f.read()
        return data

    def _process(self) -> None:
        chars = sorted(list(set(self.data)))
        self._vocab_size = len(chars)

        self.ch_int = {ch: i for i, ch in enumerate(chars)}
        self.int_ch = {i: ch for i, ch in enumerate(chars)}

    def vocab_size(self) -> int:
        if not self._vocab_size:
            self._process()
        return self._vocab_size

    def encoder(self):
        if not self.ch_int:
            self._process()
        enc = lambda x: [self.ch_int[c] for c in x]
        return enc
    
    def decoder(self):
        if not self.int_ch:
            self._process()
        dec = lambda x: ''.join([self.int_ch[int(i)] for i in x])
        return dec
    
    def _train_val_split(self):
        if not self._vocab_size:
            self._process()
        n = int(len(self.data)*self.val_split)
        train_split = self.data[:n]
        val_split = self.data[n:]
        enc = self.encoder()
        self._train_data = jnp.array(enc(train_split))
        self._val_data = jnp.array(enc(val_split))
    
    def get_batch(self, rng: jax.Array, batch_size: int, seq_len: int, split: str = "train") -> tuple[list[int], list[int]]:
        '''
        get_batch fetches the next `batch_size` of examples, each of `seq_len` in length.
        Each example is chosen randomly from the dataset.
        Use `split` to choose between the training set and the validation set
        '''
        if self._train_data is None:
            self._train_val_split()

        data = {"train": self._train_data, "val": self._val_data}.get(split, None)
        if data is None:
            raise Exception("Split must be one of train or val")

        size = len(data)
        offsets = jax.random.randint(rng, (batch_size,), 0, size-seq_len)
        xs = get_slice(data, offsets, seq_len)
        ys = get_slice(data, offsets+1, seq_len)
        return xs, ys
    
@partial(vmap, in_axes=(None, 0, None), out_axes=0)
def get_slice(arr: jax.Array, offset: jax.Array, seq_len: int):
  return lax.dynamic_slice(arr, (offset,), (seq_len,))
    



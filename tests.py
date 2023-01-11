import jax
import jax.numpy as jnp
import numpy as np

from model import (
    GPTParams,
    LayerNormParams,
    MultiHeadAttentionParams,
    PositionWiseFFNParams,
    TransformerBlockParams,
    gpt,
)


def initialize_gpt_params(
    d_model: int,
    d_ff: int,
    vocab_size: int,
    n_layers: int,
    max_seq_len: int,
    seed: int,
):
    # use numpy.random instead of jax.random to avoid using the clunky jax.random keys
    rng = np.random.RandomState(seed)

    # initialize params
    # - weights matrixes and embeddings are initialized using a normal distribution
    # - biases are initialized as zero
    # - layer norm gamma is initialized with ones, beta is initialized with zeros
    params = GPTParams(
        W_e=rng.normal(size=(vocab_size, d_model)),
        W_p=rng.normal(size=(max_seq_len, d_model)),
        blocks=[
            TransformerBlockParams(
                mha_params=MultiHeadAttentionParams(
                    W_qkv=rng.normal(size=(d_model, 3 * d_model)),
                    b_qkv=np.zeros(shape=(3 * d_model,)),
                    W_out=rng.normal(size=(d_model, d_model)),
                    b_out=np.zeros(shape=(d_model,)),
                ),
                ffn_params=PositionWiseFFNParams(
                    W1=rng.normal(size=(d_model, d_ff)),
                    b1=np.zeros(shape=(d_ff,)),
                    W2=rng.normal(size=(d_ff, d_model)),
                    b2=np.zeros(shape=(d_model,)),
                ),
                ln_1_params=LayerNormParams(
                    gamma=np.ones(shape=(d_model,)), beta=np.zeros(shape=(d_model,))
                ),
                ln_2_params=LayerNormParams(
                    gamma=np.ones(shape=(d_model,)), beta=np.zeros(shape=(d_model,))
                ),
            )
            for _ in range(n_layers)
        ],
        ln_f=LayerNormParams(gamma=np.ones((d_model,)), beta=np.zeros((d_model,))),
    )

    # convert numpy arrays to jax arrays to make run time type checker happy
    params = jax.tree_map(jnp.array, params)

    return params


def test_gpt():
    params = initialize_gpt_params(
        d_model=512,
        d_ff=512 * 4,
        vocab_size=50000,
        n_layers=12,
        max_seq_len=1024,
        seed=123,
    )
    input_ids = jnp.array([100, 12, 0, 10, 63, 2513.0])  # some random ids
    output = gpt(input_ids, params, h=8)
    assert output.shape == (len(input_ids), 50000)

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, UInt

PRNGKeyType = UInt[Array, "2"]


##############
#### GeLU ####
##############
def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


#################
#### Softmax ####
#################
def softmax(x, axis=-1):
    exp_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)


####################
#### Layer Norm ####
####################
class LayerNormParams(NamedTuple):
    gamma: Float[Array, "d_model"]
    beta: Float[Array, "d_model"]


def layer_norm(
    x: Float[Array, "seq_len d_model"], params: LayerNormParams, eps: float = 1e-5
) -> Float[Array, "seq_len d_model"]:
    mean = jnp.mean(x, axis=-1)[..., None]
    variance = jnp.var(x, axis=-1)[..., None]
    out = (x - mean) / jnp.sqrt(variance + eps)
    return params.gamma * out + params.beta


###########################
#### Position-wise FFN ####
###########################
class PositionWiseFFNParams(NamedTuple):
    W1: Float[Array, "d_model d_ff"]
    b1: Float[Array, "d_ff"]
    W2: Float[Array, "d_ff d_model"]
    b2: Float[Array, "d_model"]


def position_wise_ffn(
    x: Float[Array, "seq_len d_model"], params: PositionWiseFFNParams
) -> Float[Array, "seq_len d_model"]:
    return gelu(x @ params.W1 + params.b1) @ params.W2 + params.b2


###################
#### Attention ####
###################
class MultiHeadAttentionParams(NamedTuple):
    W_qkv: Float[Array, "d_model 3*d_model"]
    b_qkv: Float[Array, "3*d_model"]
    W_out: Float[Array, "d_model d_model"]
    b_out: Float[Array, "d_model"]


def attention(
    Q: Float[Array, "n_q d_k"],
    K: Float[Array, "n_k d_k"],
    V: Float[Array, "n_k d_v"],
    mask: Bool[Array, "n_q n_k"],
) -> Float[Array, "n_q d_v"]:
    d_k = K.shape[-1]
    attn = Q @ K.T / jnp.sqrt(d_k)
    attn = mask * attn - 1e10 * (1 - mask)
    attn = softmax(attn)
    return attn @ V


def multi_head_casual_self_attention(
    x: Float[Array, "seq_len d_model"], params: MultiHeadAttentionParams, h: int
) -> Float[Array, "seq_len d_model"]:
    # qkv projection
    # [seq_len, d_model] -> [seq_len, 3*d_model]
    x = x @ params.W_qkv + params.b_qkv

    # split into qkv
    # [seq_len, 3*d_model] -> 3 of [seq_len, d_model]
    q, k, v = jnp.split(x, 3, axis=-1)

    # split into heads
    # 3 of [seq_len, d_model] -> 3 of [h, seq_len, d_model/h]
    q, k, v = map(lambda x: jnp.array(jnp.split(x, h, axis=-1)), [q, k, v])

    # casual mask
    seq_len = q.shape[1]
    casual_mask = jnp.tri(seq_len, dtype=bool)  # [seq_len, seq_len]

    # perform attention for each head
    attn_fn = jax.vmap(attention, in_axes=(0, 0, 0, None))
    heads = attn_fn(q, k, v, casual_mask)  # [h, seq_len, d_model/h]

    # merge heads
    # [h, seq_len, d_model/h] -> [seq_len, d_model]
    x = jnp.hstack(heads)

    # output projection
    # [seq_len, d_model] -> [seq_len, d_model]
    return x @ params.W_out + params.b_out


###########################
#### Transformer Block ####
###########################
class TransformerBlockParams(NamedTuple):
    mha_params: MultiHeadAttentionParams
    ffn_params: PositionWiseFFNParams
    ln_1_params: LayerNormParams
    ln_2_params: LayerNormParams


def transformer_block(
    x: Float[Array, "seq_len d_model"], params: TransformerBlockParams, h: int
) -> Float[Array, "seq_len d_model"]:
    mha = lambda x: multi_head_casual_self_attention(x, params.mha_params, h)
    ffn = lambda x: position_wise_ffn(x, params.ffn_params)
    ln_1 = lambda x: layer_norm(x, params.ln_1_params)
    ln_2 = lambda x: layer_norm(x, params.ln_2_params)

    x = x + mha(ln_1(x))
    x = x + ffn(ln_2(x))

    return x


#############
#### GPT ####
#############
class GPTParams(NamedTuple):
    W_e: Float[Array, "vocab_size d_model"]
    W_p: Float[Array, "max_seq_len d_model"]
    blocks: list[TransformerBlockParams]
    ln_f: LayerNormParams


def gpt(
    input_ids: Int[Array, "seq_len"], params: GPTParams, h: int
) -> Float[Array, "seq_len vocab_size"]:
    # token + position embeddings
    # [seq_len] -> [seq_len, d_model]
    seq_len = input_ids.shape[0]
    x = params.W_e[input_ids] + params.W_p[jnp.arange(seq_len)]

    # feed forward through transformer blocks
    for block in params.blocks:
        # [seq_len, d_model] -> [seq_len, d_model]
        x = transformer_block(x, block, h)

    # projection to vocab
    # [seq_len, d_model] -> [seq_len, vocab_size]
    x = layer_norm(x, params.ln_f)
    return x @ params.W_e.T

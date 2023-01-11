from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from tqdm import tqdm

from model import GPTParams, PRNGKeyType, gpt, softmax


def generate(
    input_ids: Int[Array, "seq_len"],
    params: GPTParams,
    h: int,
    n_tokens_to_generate: int,
    sample_fn: Callable[[Float[Array, "vocab_size"], PRNGKeyType], int],
    seed: int,
) -> Int[Array, "n_tokens_to_generate"]:
    max_seq_len = params.W_p.shape[0]
    assert len(input_ids) + n_tokens_to_generate <= max_seq_len

    # keep decoding until either we reach generate_seq_len
    key = jax.random.PRNGKey(seed)
    for _ in tqdm(range(n_tokens_to_generate), desc="generating tokens"):
        # feed forward
        logits = gpt(input_ids, params, h)

        # get prob dist prediction for next token
        next_word_prob_dist = softmax(logits[-1])

        # sample from the probability dist to get the next token
        key, subkey = jax.random.split(key)
        next_token_id = sample_fn(next_word_prob_dist, subkey)

        # add the predicted token to our input
        input_ids = jnp.append(input_ids, next_token_id)

    # only return newly generated tokens
    original_seq_len = len(input_ids) - n_tokens_to_generate
    output_ids = input_ids[original_seq_len:]
    return output_ids


def greedy_sample(p: Float[Array, "vocab_size"], key=PRNGKeyType) -> int:
    return jnp.argmax(p)


def categorical_sample(p: Float[Array, "vocab_size"], key=PRNGKeyType) -> int:
    return jax.random.choice(key, a=len(p), p=p)

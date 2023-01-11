import argparse
import json
import os
import random
import re
from typing import Optional

import jax.numpy as jnp
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder
from model import (
    GPTParams,
    LayerNormParams,
    MultiHeadAttentionParams,
    PositionWiseFFNParams,
    TransformerBlockParams,
)
from sampling import categorical_sample, generate, greedy_sample


def download_gpt2_files(model_type: str, model_dir: str):
    assert model_type in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        r = requests.get(
            "https://openaipublic.blob.core.windows.net/gpt-2/"
            + "models"
            + "/"
            + model_type
            + "/"
            + filename,
            stream=True,
        )
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path: str, hparams: dict) -> GPTParams:
    # load tf var names and arrays from checkpoint
    init_vars = tf.train.list_variables(tf_ckpt_path)
    top_level_params = {}
    block_params = [{} for _ in range(hparams["n_layer"])]
    for name, _ in init_vars:
        array = jnp.array(tf.train.load_variable(tf_ckpt_path, name)).squeeze()

        name = name.removeprefix("model/")
        if name.startswith("h"):  # block param
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            block_params[n][sub_name] = array
        else:  # top level param
            top_level_params[name] = array

    gpt_params = GPTParams(
        W_e=top_level_params["wte"],
        W_p=top_level_params["wpe"],
        blocks=[
            TransformerBlockParams(
                mha_params=MultiHeadAttentionParams(
                    W_qkv=params["attn/c_attn/w"],
                    b_qkv=params["attn/c_attn/b"],
                    W_out=params["attn/c_proj/w"],
                    b_out=params["attn/c_proj/b"],
                ),
                ffn_params=PositionWiseFFNParams(
                    W1=params["mlp/c_fc/w"],
                    b1=params["mlp/c_fc/b"],
                    W2=params["mlp/c_proj/w"],
                    b2=params["mlp/c_proj/b"],
                ),
                ln_1_params=LayerNormParams(
                    gamma=params["ln_1/g"], beta=params["ln_1/b"]
                ),
                ln_2_params=LayerNormParams(
                    gamma=params["ln_2/g"], beta=params["ln_2/b"]
                ),
            )
            for params in block_params
        ],
        ln_f=LayerNormParams(
            gamma=top_level_params["ln_f/g"],
            beta=top_level_params["ln_f/b"],
        ),
    )

    return gpt_params


def main(
    prompt: Optional[str],
    models_dir: str,
    model_type: str,
    n_tokens_to_generate: Optional[int],
    greedy: bool,
    seed: Optional[int],
):
    assert model_type in ["124M", "355M", "774M", "1558M"]

    # download the model files if necessary
    model_dir = os.path.join(models_dir, model_type)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        download_gpt2_files(model_type, model_dir)

    # load hparams
    with open(os.path.join(model_dir, "hparams.json")) as file:
        hparams = json.load(file)

    # load bpe tokenizer
    tokenizer = get_encoder(model_type, models_dir)

    # load the tf weights into our GPTParams object
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    # model inputs
    if prompt is not None:  # conditioned generation
        input_ids = jnp.array(tokenizer.encode(prompt))
    else:  # unconditional generation
        input_ids = jnp.array([tokenizer.encoder["<|endoftext|>"]])

    # generate
    output_ids = generate(
        input_ids=input_ids,
        params=params,
        h=hparams["n_head"],
        n_tokens_to_generate=n_tokens_to_generate
        if n_tokens_to_generate is not None
        else hparams["n_ctx"] - len(input_ids),  # generate to max sequence length
        sample_fn=greedy_sample if greedy else categorical_sample,
        seed=seed if seed is not None else random.Random(seed).randint(0, 2**32),
    )

    # decode model output to string
    output_text = tokenizer.decode(output_ids.tolist())

    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run GPT using the original openai gpt-2 weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input text to condition the outputs. If not set, we'll generate unconitioned samples using the <|endoftext|> token.",
        default=None,
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Base directory for the model directories.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="124M",
        help="Model type. Must be one of ['124M', '355M', '774M', '1558M']",
    )
    parser.add_argument(
        "--n_tokens_to_generate",
        type=int,
        default=None,
        help="Number of tokens to generate. If None, we decode the models max seq len.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="If set, greedily take token with highest probability instead of sampling from the output distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed used to sample the next token from the the model's outputs. If None, a random seed is chosen. This option has no effect when using --greedy.",
    )
    args = parser.parse_args()

    print(main(**args.__dict__))

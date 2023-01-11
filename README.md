# GPT JAX
A stupidly simple [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) implementation in [JAX](https://github.com/google/jax).

* `encoder.py`: BPE tokenizer ripped straight from [`openai/gpt-2/src/encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
* `sampling.py`: Sampling code to generate full sentences from a GPT model.
* `model.py`: The GPT model code.
* `main.py`: CLI to generate text using the official GPT-2 weights from OpenAI with the `model.py` implementation.
* `tests.py`: Unit tests.

### Dependencies
```bash
poetry install
```

For GPU support, run the following after `poetry install`:
```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

This code was tested on an M1 MacBook Pro using Python 3.9.10.

### Usage
```bash
poetry python main.py \
    --prompt "Alan Turing theorized that computers would one day become" \
    --model_type "124M" \
    --n_tokens_to_generate 40 \
    --greedy
```
Which gives the result:
```
the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```
For a full list of possible options, see `poetry run python main.py --help`.

### Tests
The `tests.py` runs a very basic test of our model: Initializes with random parameters, do a forward pass on some random inputs, and then verify the output has the correct shape. The tests can be run with `pytest`:
```bash
poetry run pytest tests.py
```
Because we use [`jaxtyping`](https://github.com/google/jaxtyping) hints in our code, we can validate our types during run-time using [`typeguard`](https://github.com/agronholm/typeguard). To enable this, we can run pytest with the following flags:
```python
poetry run pytest --jaxtyping-packages model,typeguard.typechecked tests.py
```


### Correctness
This implementation should match the results obtained from Open AI's official [`gpt-2 repo`](https://github.com/openai/gpt-2).

For example, running their code using this [`Dockerfile`](https://gist.githubusercontent.com/jaymody/9054ca64eeea7fad1b58a185696bb518/raw/Dockerfile) and set of commands (Note: This does not work on M1 Macs unfortunately):
```bash
docker build -t "openai-gpt-2" "https://gist.githubusercontent.com/jaymody/9054ca64eeea7fad1b58a185696bb518/raw/Dockerfile"
docker run -dt "openai-gpt-2" --name "openai-gpt-2-app"
docker exec -it "openai-gpt-2" /bin/bash -c 'python3 src/interactive_conditional_samples.py --length 40 --model_type 124M --top_k 1'
# paste "Alan Turing theorized that computers would one day become" when prompted
```

We get the same generation from the example in the usage section:
```
the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

Of course, there will be small small difference in the logit values due to slight differences in numerical operations between tensorflow and jax, but this should be negligible.

### Differences
This implementation is significantly slower than OpenAI's and a couple of features are left out. My goal with this implementation is to create a very simple, readable, and hackable implementation for learning, not to make it blazingly fast or feature-rich. However, here's a summary of what's missing:

1. OpenAI caches the key and value vectors for past inputs during generation. Because GPTs are casual (the output for the ith input depends only on previous inputs), during generation, the outputs and intermediate activations for all the tokens except the most recently appended one will be the same as the previous forward passes. This is a lot of unnecessary recomputation during each forward pass in `generate`. Since attention is the only place where the inputs are allowed to share information, if we just cache the key-value vectors for already seen inputs, and truncate the input to our GPT to be only the most recent token that was last predicted, we greatly reduce the inference time. This is what the variable `past` is used for in OpenAI's implementation. I left this out because it adds a lot more complexity.
2. We don't do any kind of batching, which would allow us to do multiple generations in parallel (either for the same prompt, or for multiple different prompts). Luckily, making our model work with batches is as simple as `jax.vmap(gpt, in_axes=(0, None, None))`. Gotta love jax! However, you'd need to implement some complex padding logic, and the APIs would change quite a bit to adjust for this.
3. We do not implement top-p, top-k, and temperature. I'll probably add this later at some point.

Also, I avoided the use of `jax.jit` because our input increases in length during generation, which would make `jax.jit` recompile the model for each decode step. You can solve this by padding the inputs to the length we want to generate to, but to keep this implementation, I left this optimization out.

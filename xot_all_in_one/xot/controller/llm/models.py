# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from openai import OpenAI

completion_tokens = prompt_tokens = 0
# TODO
_model = ""
_api_key=""
_base_url = ""


def chatgpt(prompt, instruct=None, model=_model, temperature=0.0, max_tokens=1000, n=1, stop=None) -> list:
    if instruct is not None:
        messages = [{"role": "system", "content": instruct}, {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    return gpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)


def gpt(messages, model, temperature, max_tokens, n, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    client = OpenAI(api_key=_api_key, base_url=_base_url)
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        n=n,
    )
    outputs.extend([choice.message.content for choice in res.choices])
    # log completion tokens
    completion_tokens += res.usage.completion_tokens
    prompt_tokens += res.usage.prompt_tokens
    return outputs

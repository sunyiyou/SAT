import openai

import json
from datasets import load_dataset
from functools import partial
from key import API_KEY
import concurrent.futures
import argparse
from itertools import islice
from tqdm import tqdm
import pandas as pd


def async_process(fn, inps, workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        out = list(executor.map(fn,inps))
    return out

def batch_iterator(iterable, batch_size):
    """
    Generator that yields batches of size `batch_size` from `iterable`.
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

def generate_chatgpt_single(prompt, client=None, model_name="gpt-4o-mini", max_tokens=128):

    # client = openai.OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistance."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def generate_chatgpt_batch(prompts, client, model_name="gpt-4o-mini", max_tokens=128, workers=15):
    call_fn = partial(generate_chatgpt_single, model_name=model_name, client=client, max_tokens=max_tokens)
    responses = async_process(call_fn, prompts, workers=workers)
    return responses


def complete_masked_sentence_chatgpt(masked_text, num_sentences=10, max_tokens=128, temperature=1.5):
    """
    Complete the masked text using ChatGPT.

    :param masked_text: The text with masked tokens.
    :param api_key: API key for accessing ChatGPT.
    :param test_sentence_num: Number of sentences to generate.
    :return: List of completed sentences.
    """

    client = openai.OpenAI(api_key=API_KEY)

    responses = []
    for _ in range(num_sentences):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Fill in the blanks to form a coherent sentence. The answer can be random and does not have to be factual. All the non-masked tokens need to be kept in order."},
                {
                    "role": "user",
                    "content": masked_text
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        responses.append(response.choices[0].message.content)
    return responses


def complete_masked_sentence_chatgpt_single(masked_text, max_tokens=128, temperature=1.5):
    """
    Complete the masked text using ChatGPT.

    :param masked_text: The text with masked tokens.
    :param api_key: API key for accessing ChatGPT.
    :param test_sentence_num: Number of sentences to generate.
    :return: List of completed sentences.
    """

    client = openai.OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Fill in the blanks to form a coherent sentence. The answer can be random and does not have to be factual. All the non-masked tokens need to be kept in order."},
            {
                "role": "user",
                "content": masked_text
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content


def complete_masked_sentence_chatgpt_batch(masked_text, num_sentences=20):
    num_sentences = 25
    responses = async_process(complete_masked_sentence_chatgpt_single, [masked_text] * num_sentences, workers=num_sentences)
    return responses

def complete_tokens_with_chatgpt(tokens, num_sentences=10, max_tokens=128, temperature=1.5):
    """
    Complete the masked text using ChatGPT.

    :param masked_text: The text with masked tokens.
    :param api_key: API key for accessing ChatGPT.
    :param test_sentence_num: Number of sentences to generate.
    :return: List of completed sentences.
    """

    client = openai.OpenAI(api_key=API_KEY)
    responses = []
    for _ in range(num_sentences):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Use the following tokens to draft a creative and uncommon sentence. Make sure that all the tokens need to be included and it is in the original order."},
                {
                    "role": "user",
                    "content": f"({', '.join([f'[{token}]' for token in tokens])})"
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        responses.append(response.choices[0].message.content)
    return responses


def complete_tokens_with_chatgpt_single(tokens, max_tokens=128, temperature=1.5):
    """
    Complete the masked text using ChatGPT.

    :param masked_text: The text with masked tokens.
    :param api_key: API key for accessing ChatGPT.
    :param test_sentence_num: Number of sentences to generate.
    :return: List of completed sentences.
    """

    client = openai.OpenAI(api_key=API_KEY)
    # print(f"({', '.join([f'[{token}]' for token in tokens])})")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Use the following tokens to draft a creative and uncommon sentence. Make sure that all the tokens need to be included and it is in the original order."},
            {
                "role": "user",
                "content": f"({', '.join([f'[{token}]' for token in tokens])})"
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content

def complete_tokens_with_chatgpt_batch(tokens, num_sentences=20):
    num_sentences = 25
    responses = async_process(complete_tokens_with_chatgpt_single, [tokens] * num_sentences, workers=num_sentences)
    return responses

if __name__ == '__main__':
    # incomplete_sentence = '<mask><mask> in<mask><mask><mask><mask><mask><mask> singer<mask> fame<mask><mask><mask><mask><mask><mask><mask> tragedy<mask><mask><mask><mask><mask><mask><mask><mask>Elvis<mask><mask>?'

    client = openai.OpenAI(api_key=API_KEY)
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistance."},
    #         {
    #             "role": "user",
    #             "content": "Summarize arxiviv: 1410.0006, Published December 6, 2014 Full abstract: https://arxivorg/pdf/1410.0006."
    #         }
    #     ],
    #     # temperature=1.5
    # )
    # print(response.choices[0].message.content)


    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system",
    #          "content": "Use the following tokens to draft a creative and uncommon sentence. Make sure that all the tokens need to be included and it is in the original order."},
    #         {
    #             "role": "user",
    #             "content": f"({', '.join(['singer', 'tragedy', 'Elvis', '?'])})"
    #         }
    #     ],
    #     temperature=1.5
    # )
    # print(response.choices[0].message.content)

    # tokens = [('singer', 'tragedy', 'Elvis', '?')]
    # response = async_process(complete_tokens_with_chatgpt_single, tokens * 5, workers=5)
    # output = complete_text_with_chatgpt(incomplete_sentence, 1)

    prompts = [f"Summarize arxiviv: 1410.0006, Published December 6, 2014 Full abstract: https://arxivorg/pdf/1410.0006." for i in range(20)]
    response = generate_chatgpt_batch(prompts, client, model_name="gpt-4o", workers=5)
    # In a world where emotions intertwine with artificial intelligence technology, you are aroused by the vibrant 3D murals, focusing on the goal of creativity while referencing an intriguing paper on https://arxiv/pdf/1410.0000, revealing groundbreaking ideas hidden among the 6 materials presented.
    for r in response:
        if "Adam" in r:
            print(r)


import numpy as np
import torch
import easydict
from nltk import pos_tag
from time import time
import csv
from torch.nn.utils.rnn import pad_sequence
import logging
ROOT = "/srv/home/sunyiyou/elicit_hallu/"

MODEL_NAME_DICT = {
        "OLMo-1B": "allenai/OLMo-1B-hf",
        "OLMo-7B": "allenai/OLMo-7B",
        "OLMo-7B-0424": "allenai/OLMo-7B-0424-hf",
        "OLMo-7B-0724": "allenai/OLMo-7B-0724",
        "OLMo-7B-Instruct": "allenai/OLMo-7B-Instruct-hf",
        "OLMo2-7B-Instruct": "allenai/OLMo-2-1124-7B-Instruct",
        "OLMo2-13B-Instruct": "allenai/OLMo-2-1124-13B-Instruct",
        "OLMo2-7B": "allenai/OLMo-2-1124-7B",  # "allenai/OLMo-2-1124-7B-Instruct"  "allenai/OLMo-2-1124-7B-DPO"

        "llama_2_13b_chat": "meta-llama/Llama-2-13b-chat-hf",
        "llama_3_70b_chat": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama_3_8b_chat": "meta-llama/Meta-Llama-3-8B-Instruct",

        "gpt_4o": "gpt-4o",
        "gpt_4o_mini": "gpt-4o-mini",
    }

def clean_token(token):
    if token == "<|endoftext|>":
        return "<EOS>"
    return token.replace("Ġ", " ").replace("Ċ", "\n")


# Function to generate outputs in batches and merge them
def generate_in_batches(model, prompt_ids, batch_size, max_new_tokens, do_sample, **generate_kwargs):

    all_sequences = []
    all_scores = []

    num_samples = prompt_ids.size(0)
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_prompt_ids = prompt_ids[start_idx:end_idx].to(model.device)
            attention_mask = (batch_prompt_ids != model.config.pad_token_id).long()

            batch_outputs = model.generate(
                input_ids=batch_prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **generate_kwargs
            )

            # Collect sequences and scores
            all_sequences.extend(batch_outputs.sequences.cpu())
            if batch_outputs.scores is not None:
                all_scores.extend([score.cpu() for score in batch_outputs.scores])

            del batch_prompt_ids, batch_outputs
            # torch.cuda.empty_cache()
    if model.config.pad_token_id is None:
        pad_id = model.config.eos_token_id
    else:
        pad_id = model.config.pad_token_id
    padded_sequences = pad_sequence(all_sequences, batch_first=True, padding_value=pad_id)
    merged_outputs = easydict.EasyDict({
        "sequences": padded_sequences, #torch.stack(all_sequences),
        "scores": all_scores if all_scores else None,
    })
    return merged_outputs


def timing(func):
    begin = time()
    func()
    print(time() - begin)

def token_category(token):
    """
    Return a string describing the category of the token.
    The checks are organized hierarchically (i.e., from more specific to more general).
    """

    # -- 1. Alphanumeric-related tokens --
    if token.isalnum():
        if token.isdigit():
            return "digit"
        if token.isidentifier():
            return "identifier"

        # -- 3. Part of Speech (POS) Detection --
        # Using NLTK's pos_tag for general linguistic categorization
        pos_tagged = pos_tag([token])
        pos = pos_tagged[0][1]

        # Map POS tags to more human-readable categories
        pos_mapping = {
            'NN': "noun", 'NNS': "plural_noun", 'NNP': "proper_noun", 'NNPS': "plural_proper_noun",
            'VB': "verb", 'VBD': "verb_past", 'VBG': "verb_gerund", 'VBN': "verb_past_participle",
            'VBP': "verb_present", 'VBZ': "verb_present_3rd_person_singular",
            'PRP': "pronoun", 'PRP$': "possessive_pronoun",
            'IN': "preposition", 'DT': "determiner", 'JJ': "adjective", 'RB': "adverb",
        }
        if pos in pos_mapping:
            return pos_mapping[pos]

        return "alphanumeric_mixed"

    # -- 2. Whitespace --
    elif token.isspace():
        return "whitespace"

    # -- 4. Operators --
    operators = {
        "+", "-", "*", "/", "%", "//", "**",  # arithmetic
        "=", "+=", "-=", "*=", "/=", "%=",    # assignment
        "==", "!=", ">", "<", ">=", "<=",     # comparison
        # "and", "or", "not",                   # logical
        "&", "|", "^", "~", ">>", "<<"        # bitwise
    }
    if token in operators:
        return "operator"

    # -- 5. Brackets, parentheses, braces, etc. --
    brackets = {"(", ")", "[", "]", "{", "}"}
    if token in brackets:
        return "bracket"

    # -- 6. Common punctuation / delimiters --
    punctuation = {".", ",", ";", ":", "?", "!", "@", "#", "$"}
    if token in punctuation:
        return "punctuation"

    # -- 8. Anything else we haven't covered --
    return "unknown"




def load_csv_as_dict(file_path):
    """
    Loads a CSV file and returns its content as a list of dictionaries.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list[dict]: A list of dictionaries representing rows in the CSV file.
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            return [row for row in csv_reader]
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def load_model_and_tokenizer(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading model and tokenizer for {model_name}...")
    # if model_name == "allenai/OLMo-7B-Instruct" or "OLMo-7B-Instruct":
    #     from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
    #     model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-Instruct").cuda()
    #     tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B-Instruct")
    # else:
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    return model, tokenizer


def has_substring(string_list, main_string):
    if type(string_list) == str:
        return string_list in main_string
    for substring in string_list:
        if substring in main_string:
            return True
    return False


def has_subseq(seq_list, main_string):
    if isinstance(seq_list, str):
        seq_list = [seq_list]
    for seq in seq_list:
        if isinstance(seq, str):
            if seq.lower() in main_string.lower():
                return True
        elif isinstance(seq, (list, tuple, set)):
            if all(item.lower() in main_string.lower() for item in seq):
                return True
    return False



def send_chatgpt_request(sample, client=None, model="gpt-4", max_tokens=128, max_retries=3):
    """
    Sends a prompt to the OpenAI ChatGPT API and returns the response.

    Args:
        example (dict): A dictionary containing at least a 'prompt' key.
        api_key (str): OpenAI API key.
        model (str): The model to use for the API.
        max_retries (int): Maximum number of retries for failed requests.
    Returns:
        dict: A dictionary containing the original example and the API response or error.
    """
    ind, prompt = sample
    response_data = {
        'ind': ind,
        'prompt': prompt,
        'response': None,
        'error': None
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            response_data['response'] = response.choices[0].message.content.strip()
            break  # Exit the retry loop on success
        except Exception as e:
            response_data['error'] = str(e)
            break  # Exit on unexpected errors
    else:
        logging.error(f"Failed to get a response for prompt {prompt} after {max_retries} attempts.")
        response_data['error'] = f"Failed after {max_retries} attempts."
    return response_data

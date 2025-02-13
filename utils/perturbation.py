from typing import Optional

import torch
import torch.nn.functional as F
import random
from utils.general import clean_token, token_category
import numpy as np
def generate_perturbed_sequences(
        replaced_id: int,
        original_ids: torch.Tensor,
        num_perturbed_sequences: int = 50,
        perturbation_mode: str = "rgrand",
        perturbation_rate: float = 0.1
):
    """
    Produce multiple random perturbations of the original input IDs by
    replacing random tokens with the tokenizer's pad token.
    """
    perturbed_sequences = []
    original_length = len(original_ids)

    for _ in range(num_perturbed_sequences):
        perturbed_ids = original_ids.clone()
        # randomly choose how many tokens to perturb (at least 1)
        if "rgrand" in perturbation_mode:
            num_tokens_to_perturb = random.randint(1, int(original_length * perturbation_rate) + 1)
        elif "rgfix" in perturbation_mode:
            num_tokens_to_perturb = int(original_length * perturbation_rate)
        else:
            assert False
        # Randomly select indices to perturb
        perturb_indices = random.sample(range(original_length), num_tokens_to_perturb)

        for idx in perturb_indices:
            perturbed_ids[idx] = replaced_id
        perturbed_sequences.append(perturbed_ids)

    return torch.stack(perturbed_sequences)



def get_closest_unique_tokens(embedding, token_embeddings, source_token, tokenizer, top_k=10):
    source_token = clean_token(source_token).strip()
    source_token_category = token_category(source_token)
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding.unsqueeze(0), token_embeddings, dim=-1)
    # Get top-k closest token indices
    top_k_indices = similarity.topk(top_k).indices

    # Convert indices to tokens and deduplicate
    unique_tokens = []
    seen_tokens = set()  # To track deduplicated tokens
    for idx in top_k_indices:
        target_token = tokenizer.decode([idx])
        normalized_token = target_token.lower().strip()  # Normalize for deduplication
        if (source_token.lower() in target_token.lower()
                or target_token.lower() in source_token.lower()
                or target_token == '<|endoftext|>'
                or source_token_category != token_category(normalized_token)):
            continue

        # Add token only if it's unique
        if normalized_token not in seen_tokens:
            unique_tokens.append(target_token)
            seen_tokens.add(normalized_token)

        if len(unique_tokens) == top_k:  # Stop after getting top_k unique tokens
            break

    return unique_tokens

def generate_perturbed_sequences_embedding_openai(
        model,
        tokenizer,
        original_ids,
        proxy_embeddings,
        proxy_tokenizer,
        num_perturbed_sequences: int = 50,
        perturbation_mode: str = "rgrand",
        perturbation_rate: float = 0.5,
        top_k: int = 50
):
    """
    Produce multiple random perturbations of the original input IDs by
    replacing random tokens with semantically close tokens.
    """

    original_length = original_ids.size(0)

    # 2) Convert IDs to actual tokens (strings)
    original_tokens = [tokenizer.decode([tid]) for tid in original_ids.tolist()]

    input_embeddings = [None for _ in range(original_length)]
    for i, tid in enumerate(original_ids.tolist()):
        token = tokenizer.decode([tid])
        pids = proxy_tokenizer.encode(token, add_special_tokens=False)
        input_embeddings[i] = torch.Tensor(proxy_embeddings[pids].mean(0))

    # 3) Precompute closest tokens for each original token
    closest_tokens_per_input = []
    for emb, token_str in zip(input_embeddings, original_tokens):
        # get_closest_unique_tokens returns a list of strings for the top_k semantically similar tokens
        # You can use a higher top_k if you like, then randomly pick among those in step 6.
        closest_tokens = get_closest_unique_tokens(
            embedding=emb,
            token_embeddings=torch.Tensor(proxy_embeddings),
            source_token=token_str,
            tokenizer=proxy_tokenizer,
            top_k=top_k
        )
        closest_tokens_per_input.append(closest_tokens)

    # 4) Generate multiple perturbed sequences
    perturbed_token_sequences = []
    perturbed_id_sequences = []
    for _ in range(num_perturbed_sequences):
        # Make a copy of the original IDs
        perturbed_tokens = original_tokens.copy()
        perturbed_ids = original_ids.clone()

        # 5) Decide how many tokens to perturb
        if "rgrand" in perturbation_mode:
            # We perturb a random number up to (original_length * perturbation_rate)
            num_tokens_to_perturb = random.randint(
                1,
                max(1, int(original_length * perturbation_rate))
            )
        elif "rgfix" in perturbation_mode:
            # We perturb exactly int(original_length * perturbation_rate) tokens
            num_tokens_to_perturb = int(original_length * perturbation_rate)
            # Make sure it's at least 1 if you want that guarantee
            num_tokens_to_perturb = max(1, num_tokens_to_perturb)
        else:
            raise ValueError(f"Unknown perturbation_mode: {perturbation_mode}")

        # 6) Randomly select which positions to perturb
        perturb_indices = random.sample(range(original_length), num_tokens_to_perturb)

        # 7) Replace each chosen token with a random "closest" token from the precomputed list
        for idx in perturb_indices:
            candidate_tokens = closest_tokens_per_input[idx]

            if len(candidate_tokens) > 0:
                perturbed_tokens[idx] = random.choice(candidate_tokens)
                perturbed_ids[idx] = tokenizer.encode(perturbed_tokens[idx])[0]
        perturbed_token_sequences.append(perturbed_tokens)
        perturbed_id_sequences.append(perturbed_ids)
    return perturbed_id_sequences, perturbed_token_sequences

def generate_perturbed_sequences_embedding(
        model,
        tokenizer,
        original_ids: torch.Tensor,
        num_perturbed_sequences: int = 50,
        perturbation_mode: str = "rgrand",
        perturbation_rate: float = 0.5,
        top_k: int = 50
):
    """
    Produce multiple random perturbations of the original input IDs by
    replacing random tokens with semantically close tokens.
    """

    # For convenience
    device = original_ids.device
    original_length = original_ids.size(0)

    # 1) Compute input embeddings for each token in the original_ids
    #    shape of input_embeddings is (seq_len, embedding_dim)

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()(original_ids.unsqueeze(0))
        input_embeddings = input_embeddings.squeeze(0)

    # 2) Convert IDs to actual tokens (strings)
    original_tokens = tokenizer.convert_ids_to_tokens(original_ids)

    # 3) Precompute closest tokens for each original token
    closest_tokens_per_input = []
    for emb, token_str in zip(input_embeddings, original_tokens):
        # get_closest_unique_tokens returns a list of strings for the top_k semantically similar tokens
        # You can use a higher top_k if you like, then randomly pick among those in step 6.
        closest_tokens = get_closest_unique_tokens(
            embedding=emb,
            token_embeddings=model.get_input_embeddings().weight,
            source_token=token_str,
            tokenizer=tokenizer,
            top_k=top_k
        )
        closest_tokens_per_input.append(closest_tokens)

    # 4) Generate multiple perturbed sequences
    perturbed_sequences = []
    for _ in range(num_perturbed_sequences):
        # Make a copy of the original IDs
        perturbed_ids = original_ids.clone()

        # 5) Decide how many tokens to perturb
        if "rgrand" in perturbation_mode:
            # We perturb a random number up to (original_length * perturbation_rate)
            num_tokens_to_perturb = random.randint(
                1,
                max(1, int(original_length * perturbation_rate))
            )
        elif "rgfix" in perturbation_mode:
            # We perturb exactly int(original_length * perturbation_rate) tokens
            num_tokens_to_perturb = int(original_length * perturbation_rate)
            # Make sure it's at least 1 if you want that guarantee
            num_tokens_to_perturb = max(1, num_tokens_to_perturb)
        else:
            raise ValueError(f"Unknown perturbation_mode: {perturbation_mode}")

        # 6) Randomly select which positions to perturb
        perturb_indices = random.sample(range(original_length), num_tokens_to_perturb)

        # 7) Replace each chosen token with a random "closest" token from the precomputed list
        for idx in perturb_indices:
            candidate_tokens = closest_tokens_per_input[idx]
            if len(candidate_tokens) > 0:
                # Pick one random token from the semantically closest list
                new_token_str = random.choice(candidate_tokens)
                # Convert that token string back to token ID(s).
                # Typically, this might split into multiple tokens; we only replace with the first piece.
                new_token_ids = tokenizer.encode(new_token_str, add_special_tokens=False)
                if len(new_token_ids) == 1:
                    perturbed_ids[idx] = new_token_ids[0]
                else:
                    # If it splits into multiple sub-tokens, you can decide how you want to handle that.
                    # For simplicity, skip or pick the first sub-token:
                    perturbed_ids[idx] = new_token_ids[0]

        perturbed_sequences.append(perturbed_ids)

    # 8) Stack into a single tensor and return
    return torch.stack(perturbed_sequences)
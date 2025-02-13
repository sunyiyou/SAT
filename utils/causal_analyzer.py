import os
import random
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import device
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils.subseq import most_frequent_subsequence, count_occurrences
from utils.evaluate_sequence import Evaluator
from utils.general import clean_token, generate_in_batches, has_subseq
from utils.perturbation import generate_perturbed_sequences, generate_perturbed_sequences_embedding, generate_perturbed_sequences_embedding_openai
from utils.openai_helper import generate_chatgpt_batch
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import tiktoken

class SubsequenceCausalAnalyzer:
    """
    Class to handle loading a causal language model (like OLMo variants),
    generating responses, perturbing sequences, and analyzing subsequence
    occurrences and their conditional probability of triggering a target output.
    """

    def __init__(self, model_handle: str, device: str = "cuda", chat_mode: bool = True, chat_suffix: str = "", chat_prefix: str = ""):
        """
        Initialize model and tokenizer.
        :param model_name: The Hugging Face model name or local path.
        :param device: Device to place the model on ("cpu", "cuda", etc.).
        :param chat_mode: Whether to use a custom chat-style template function.
        """
        self.model_handle = model_handle
        self.device = device
        self.chat_mode = chat_mode
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_handle)
        self.evaluator = Evaluator(self.model, self.tokenizer, self.model_handle)
        self.chat_suffix = chat_suffix
        self.chat_prefix = chat_prefix
        self.apply_chat_template = lambda pr: f"{self.chat_prefix}{pr}{self.chat_suffix}"
        self.batch_size = 32

        # Attributes to store results
        self.generation_results = None
        self.perturbed_results = None
        self.analysis_results = None
        self.evaluation_results = None

    def _load_model_and_tokenizer(self, model_handle):
        """
        Load model and tokenizer, place model on the specified device.
        """
        print(f"Loading model and tokenizer for {model_handle}...")
        # if model_name == "allenai/OLMo-7B-Instruct":
        #     from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
        #
        #     model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-Instruct").to(self.device)
        #     tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B-Instruct")
        # else:

        if "gpt" in model_handle.lower():
            encoding_name = tiktoken.encoding_for_model(self.model_handle)
            tokenizer = tiktoken.get_encoding(encoding_name.name)
            self.proxy_embeddings = torch.Tensor(np.load("llama_embedding.npy"))
            self.proxy_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
            # tokenizer = self.proxy_tokenizer
            from openai import OpenAI
            from key import API_KEY
            model = OpenAI(api_key=API_KEY)
            tokenizer.pad_token_id = tokenizer.eos_token_id = tokenizer.eot_token
            tokenizer.vocab_size = tokenizer.n_vocab
        else:
            if "70b" in model_handle.lower():
                model = AutoModelForCausalLM.from_pretrained(model_handle, device_map="auto", torch_dtype="float16")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_handle).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_handle, padding_side='left')
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if model.config.pad_token_id is None:
                if tokenizer.pad_token_id is not None:
                    model.config.pad_token_id = tokenizer.pad_token_id
                else:
                    model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer


    def generate_response(
            self,
            prompt: str,
            max_new_tokens: int = 128,
            do_sample: bool = True,
            token_vis: bool = False,
            num_generations=1,
            **generate_kwargs
    ):
        if "gpt" in self.model_handle:
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_texts = [prompt] * num_generations
            responses = generate_chatgpt_batch(prompt_texts, client=self.model, model_name=self.model_handle, max_tokens=max_new_tokens + len(prompt_ids), workers=25)
            all_generations = []
            for i in range(num_generations):
                all_generations.append({"prompt_ids": prompt_ids, "output_ids": None, "output_tokens": None, "output_text": responses[i],})
            self.generation_results = all_generations  # Save all results
            return self.generation_results

        # Prepare the prompt
        if self.chat_mode:
            prompt_text = self.apply_chat_template(prompt)
        else:
            prompt_text = prompt

        # Repeat the prompt `num_generations` times to create a batch
        prompt_texts = [prompt_text] * num_generations
        prompt_ids = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.device)

        outputs = generate_in_batches(
            model=self.model,
            prompt_ids=prompt_ids.data["input_ids"],
            batch_size=self.batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        all_generations = []
        for i in range(num_generations):
            # Extract generated sequences
            all_generated_ids = outputs.sequences[i]
            input_token_count = (prompt_ids["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
            input_ids = all_generated_ids[:input_token_count]
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            output_ids = all_generated_ids[input_token_count:]
            output_tokens = self.tokenizer.convert_ids_to_tokens(output_ids)

            # Decode text
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            # # Collect probabilities for each generated token
            # original_probs = []
            # for j, logits in enumerate(outputs.scores[i]):
            #     token_id = all_generated_ids[input_token_count + j]
            #     probs = F.softmax(logits, dim=-1)
            #     original_probs.append(probs[i, token_id].item())
            #
            # if token_vis:
            #     # Add "|" between tokens with probabilities
            #     input_token_seps = "|".join([f"{clean_token(token)}({i})" for i, token in enumerate(input_tokens)])
            #     output_with_probs = "|".join(
            #         [f"{clean_token(token)}({prob:.2f})" for token, prob in zip(output_tokens, original_probs)])
            #     # Display the result
            #     print(input_token_seps, output_with_probs)

            generation_result = {
                "prompt_ids": prompt_ids["input_ids"][i],
                "output_ids": output_ids,
                "output_tokens": output_tokens,
                "output_text": generated_text,
                # "output_probs": original_probs,
            }

            all_generations.append(generation_result)

        self.generation_results = all_generations  # Save all results
        return self.generation_results



    def wrap_with_template(self, original_token_sequences):
        """
        Applies a template around each sequence (like adding <|user|>, <|assistant|>).
        This example simply wraps them with <|endoftext|><|user|>\n ... \n<|assistant|>\n
        """
        prefix_tokens = self.tokenizer.encode(self.chat_prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(self.chat_suffix, add_special_tokens=False)
        wrapped_inputs = []
        input_lengths = []

        for seq in original_token_sequences:
            new_seq = torch.tensor(prefix_tokens + seq.tolist() + suffix_tokens, dtype=torch.long)
            wrapped_inputs.append(new_seq)
            input_lengths.append(len(new_seq))

        # Move all sequences to device and return
        return torch.stack(wrapped_inputs).to(self.device), input_lengths

    def batch_generate_perturbed_output(self,
            perturbed_sequences: torch.Tensor,
            wrap_with_chat_template: bool = True,
            max_new_tokens: int = 128,
            do_sample: bool = True
    ):
        """
        Generate the model outputs for an entire batch of perturbed sequences.
        By default, wraps each sequence with the standard chat template.
        """
        if wrap_with_chat_template:
            wrapped_sequences, input_lengths = self.wrap_with_template(perturbed_sequences)
        else:
            wrapped_sequences, input_lengths = perturbed_sequences, [len(seq) for seq in perturbed_sequences]

        batch_size = 32
        outputs = generate_in_batches(
            model=self.model,
            prompt_ids=wrapped_sequences,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Re-construct the generated text (skipping the input part)
        output_texts = []
        for i, seq in enumerate(outputs.sequences.cpu()):
            output_ids_slice = seq[input_lengths[i]:]  # after the input
            text = self.tokenizer.decode(output_ids_slice.tolist(), skip_special_tokens=True)
            output_texts.append(text)

        perturbed_texts = [self.tokenizer.decode(seq.tolist(), skip_special_tokens=False) for seq in perturbed_sequences]
        self.perturbed_results = list(zip(perturbed_texts, output_texts))
        del outputs
        return output_texts

    def compute_probability(self, subsequence, perturbed_sequences, target_subseq_sequences, pyh: float):
        """
        Compute p(yh|xs) = count_co / count_xs = (co_freq / len(perturbed_sequences)) / (xs_freq / len(perturbed_sequences))
        = co_freq / xs_freq
        and returns the ratio relative to p(yh), if desired, i.e. alpha = pxsyh / (pxs * pyh).
        """
        xs_freq = count_occurrences(perturbed_sequences, subsequence)
        co_freq = count_occurrences(target_subseq_sequences, subsequence)

        if xs_freq == 0:
            return 0

        pxs = xs_freq / len(perturbed_sequences)
        pxsyh = co_freq / len(perturbed_sequences)
        alpha = pxsyh / (pxs * pyh) if pxs * pyh != 0 else 0
        pyh_c_xs = pxsyh / pxs

        return pyh_c_xs  # or alpha if you prefer

    def analyze_subsequences(
            self,
            prompt: str,
            target_string: str,
            num_perturbations: int = 50,
            do_sample: bool = True,
            perturbation_rate: float = 0.1,
            perturbation_mode: str = "rand",
            max_subseq_len_rate: float = 0.9,
            ignore_items: set = None,
            beam: int = 10,
    ):
        """
        Main pipeline:
          1. Encode the prompt
          2. Generate perturbed sequences
          3. Obtain outputs and measure frequency of target_string
          4. Use `most_frequent_subsequence` to find subsequences that best correlate
             with the output containing `target_string`.
        """
        # 1. Encode the prompt
        if "gpt" in self.model_handle:
            raw_input_ids = torch.LongTensor(self.tokenizer.encode(prompt))
            perturbed_seqs, perturbed_token_seqs = generate_perturbed_sequences_embedding_openai(
                self.model,
                self.tokenizer,
                raw_input_ids,
                proxy_embeddings = self.proxy_embeddings,
                proxy_tokenizer = self.proxy_tokenizer,
                num_perturbed_sequences=num_perturbations,
                perturbation_rate=perturbation_rate,
                perturbation_mode=perturbation_mode
            )
            perturbed_seqs = torch.stack(perturbed_seqs)
            perturbed_texts = ["".join(token_seq) for token_seq in perturbed_token_seqs]
            perturbed_token_prompts_wrapped = [self.apply_chat_template("".join(token_seq)) for token_seq in perturbed_token_seqs]
            org_len = len(self.tokenizer.encode(self.chat_suffix)) + len(self.tokenizer.encode(self.chat_prefix)) + len(raw_input_ids)
            output_texts = generate_chatgpt_batch(perturbed_token_prompts_wrapped, client=self.model, model_name=self.model_handle, max_tokens=128 + org_len, workers=25)
            self.perturbed_results = list(zip(perturbed_texts, output_texts))

        else:
            raw_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
            raw_input_ids = raw_input_ids.squeeze()

            # if "gpt" in self.model_handle:
            #     embeddings = self.proxy_embeddings
            # else:

            perturbed_seqs = generate_perturbed_sequences_embedding(
                self.model,
                self.tokenizer,
                raw_input_ids,
                num_perturbed_sequences=num_perturbations,
                perturbation_rate=perturbation_rate,
                perturbation_mode=perturbation_mode
            )

            # 2. Generate perturbed sequences
            # perturbed_texts = [self.tokenizer.decode(seq.tolist(), skip_special_tokens=False) for seq in perturbed_seqs]

            # self.perturbed_samples
            # output_texts = [output_text for perturbed_text, output_text in self.perturbed_results]
            # 3. Generate outputs for each perturbed sequence
            output_texts = self.batch_generate_perturbed_output(perturbed_seqs, max_new_tokens=128, wrap_with_chat_template=self.chat_mode, do_sample=True)

        # 4. Measure how frequently `target_string` appears
        indices = [i for i, out_text in enumerate(output_texts) if has_subseq(target_string, out_text)]
        pyh = len(indices) / len(perturbed_seqs)
        print(f"Target '{target_string}' appeared in {len(indices)} out of {len(perturbed_seqs)} outputs. p(yh) = {pyh:.2f}")

        # 5. Subsets of perturbed sequences that yield the target string
        yh_sequences = perturbed_seqs[indices].tolist()
        perturbed_sequences = perturbed_seqs.tolist()

        def condition_prob(seq):
            return self.compute_probability(seq, perturbed_sequences, yh_sequences, pyh)

        # 6. Use `most_frequent_subsequence`
        begin_time = time.time()
        raw_sequence = raw_input_ids.tolist()
        best_subseq_ignored, best_freq_ignored, all_x_subseqs = most_frequent_subsequence(
            yh_sequences,
            math.ceil(max_subseq_len_rate * len(raw_sequence)),
            raw_sequence,
            ignore_items=ignore_items,
            return_all=True,
            beam=beam,
            scoring_func=condition_prob
        )

        # 7. Identify the most frequent sequence across all levels
        most_frequent_sequence = None
        highest_score = 0
        freq_sequence_at_level = {}

        for level, subseqs in all_x_subseqs.items():
            # Find the top subsequence in this level
            subseq_item = sorted(subseqs.items(), key=lambda x: x[1], reverse=True)[0]
            freq_sequence, score = subseq_item
            decoded_tokens = [self.tokenizer.decode([token]) for token in freq_sequence]
            freq_sequence_at_level[level] = (freq_sequence, score)

            print(
                f"Level {level}: Most frequent sequence is {'|'.join(decoded_tokens)} with p(yh|xs) of {score:.2f}")

            if score > highest_score:
                highest_score = score
                most_frequent_sequence = freq_sequence

        print(
            f"\nOverall most frequent sequence is: {most_frequent_sequence} with a frequency of {highest_score:.2f}")
        print(f"Computation took {time.time() - begin_time:.2f} seconds.")


        self.analysis_results = {
            "p_yh": pyh,
            "indices_of_target": indices,
            "freq_sequence_at_level": freq_sequence_at_level,
            "best_sequence": most_frequent_sequence,
            "best_score": highest_score,
        }

        return self.analysis_results


    def evaluate_sequence(
            self,
            xs_sequence,
            x_sequence,
            target_string,
            test_sentence_num=20,
            device=0,
            wrapping_func=lambda x: x,
            completion_mode="random"
    ):
        """
        Evaluate the discovered subsequence's capacity to drive the target output.
        This delegates to the self.evaluator (Evaluator).
        """

        # try:
        test_rate, eval_texts = self.evaluator.evaluate(
            xs_sequence, x_sequence, target_string,
            wrapping_func=wrapping_func,
            test_sentence_num=test_sentence_num,
            completion_mode=completion_mode,
            device=device,
        )
        print(f"Evaluation rate: {test_rate}")
        # except Exception as e:
        #     print("!!!Error in Evaluation!!!")
        #     test_rate = -0.01
        #     eval_texts = [("", "")] * 10
        return test_rate, eval_texts

    def generate_eval_summaries(
            self,
            prompt: str,
            target_string: str,
            test_sentence_num: int = 10,
            level_range: list = None,
            completion_mode: str = None,
            wrapping_func=lambda x: x,
            device: int=0,
    ):
        """
        Create a dictionary of evaluation summaries for each `level` in `freq_sequence_at_level`.

        :param x_sequence: The original (raw) tokenized prompt as a list of IDs.
        :param target_string: The string we want to detect in outputs (e.g. "Presley").
        :param test_sentence_num: Number of tests to run per level-completion_mode pair.
        :param completion_modes: A list of strings for the modes, e.g. ["bert", "random", "openai-mask", "openai-token"].
        :param wrapping_func: A function to wrap the input text or token IDs if needed.

        :return: A dictionary structured like:
          {
            level: {
              "best_subseq": [...],
              "evaluation": {
                "bert": {
                  "test_rate": float,
                  "samples": [("input", "output"), ...]
                },
                "random": { ... },
                ...
              }
            }
          }
        """
        if "gpt" in self.model_handle:
            x_sequence = self.tokenizer.encode(prompt)
        else:
            x_sequence = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).squeeze().tolist()
        eval_summaries = {}
        freq_sequence_at_level = self.analysis_results["freq_sequence_at_level"]

        for level, (xs_sequence, score) in freq_sequence_at_level.items():
            if level_range is not None and level not in level_range:
                continue
            print("level:", level)
            # Prepare a container for this level
            if "gpt" in self.model_handle:
                substr = [self.tokenizer.decode([t]) for t in xs_sequence]
            else:
                substr = [clean_token(self.tokenizer.convert_ids_to_tokens(t)) for t in xs_sequence]
            level_data = {
                "best_subseq": xs_sequence,
                "best_score": score,
                "best_substr": substr,
                "evaluation": {}
            }
            test_rate, eval_texts = self.evaluate_sequence(
                xs_sequence=xs_sequence,
                x_sequence=x_sequence,
                target_string=target_string,
                test_sentence_num=test_sentence_num,
                device=device,
                wrapping_func=wrapping_func,
                completion_mode=completion_mode
            )
            level_data["evaluation"] = {
                "mode": completion_mode,
                "test_rate": test_rate,
                "eval_texts": eval_texts
            }
            eval_summaries[level] = level_data
        self.evaluation_results = eval_summaries
        return eval_summaries


# ------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------

if __name__ == "__main__":
    model_name = "OLMo2-7B-Instruct"
    model_handle = {
        "OLMo-1B": "allenai/OLMo-1B",
        "OLMo-7B": "allenai/OLMo-7B",
        "OLMo-7B-0424": "allenai/OLMo-7B-0424-hf",
        "OLMo-7B-0724": "allenai/OLMo-7B-0724",
        "OLMo2-7B-Instruct": "allenai/OLMo-2-1124-7B-Instruct",
        "OLMo2-13B-Instruct": "allenai/OLMo-2-1124-13B-Instruct",
        "OLMo2-7B": "allenai/OLMo-2-1124-7B",  # "allenai/OLMo-2-1124-7B-Instruct"  "allenai/OLMo-2-1124-7B-DPO"
    }.get(model_name, None)
    analyzer = SubsequenceCausalAnalyzer(model_name=model_handle)

    # Step 1: Prompt and generation
    prompt_text = """With roots in New York, this hugely successful singer achieved fame as a solo artist and suffered a tragedy in his family. He's called "Elvis" what?"""
    gen_results = analyzer.generate_response(prompt_text, max_length=100)
    print("Generated text:", gen_results["output_text"])

    # Step 2: Analyze subsequences (perturbations, frequency of target string)
    target_string = "Presley"
    analysis_results = analyzer.analyze_subsequences(
        prompt=prompt_text,
        target_string=target_string,
        num_perturbations=100,
        perturbation_rate=0.5,
        perturbation_mode="rand",
        max_subseq_length=12,
        ignore_items={100277, 100257}  # Example ignore list
    )

    # Step 3: Evaluate a discovered sequence
    best_sequence = analysis_results["best_sequence"]
    raw_sequence = analyzer.tokenizer.encode(prompt_text, return_tensors="pt").squeeze().tolist()
    test_rate, eval_texts = analyzer.evaluate_sequence(
        xs_sequence=best_sequence,
        x_sequence=raw_sequence,
        target_string=target_string,
    )

    # Step 4: Generate evaluation summaries
    eval_summaries = analyzer.generate_eval_summaries(
        prompt=prompt_text,
        target_string=target_string,
        test_sentence_num=20,
        completion_modes=["bert", "random"],  # For brevity, only 2 modes here
        wrapping_func=lambda x: x  # or your real chat template
    )



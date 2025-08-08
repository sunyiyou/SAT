from utils.bert_helper import complete_masked_sentence_bert
from utils.general import *
from utils.subseq import find_subsequence_occurrences
from utils.openai_helper import complete_tokens_with_chatgpt, complete_masked_sentence_chatgpt, complete_tokens_with_chatgpt_batch, complete_masked_sentence_chatgpt_batch, generate_chatgpt_batch
import random


def merge_consecutive_tokens(tokens, occurrence_indices):
    """
    Merges consecutive tokens (based on consecutive indices) into a single token.

    :param tokens: List of token strings.
    :param occurrence_indices: List of integer indices corresponding to each token.
    :return: A tuple (merged_tokens, merged_indices)
    """

    # If there's nothing to merge, return immediately
    if not tokens or not occurrence_indices:
        return [], []

    merged_tokens = [tokens[0]]
    merged_indices = [occurrence_indices[0]]

    for i in range(1, len(tokens)):
        current_index = occurrence_indices[i]
        previous_index = occurrence_indices[i - 1]

        if current_index == previous_index + 1:
            # If indices are consecutive, merge this token with the last merged token
            merged_tokens[-1] += tokens[i]
        else:
            # Otherwise, start a new token block
            merged_tokens.append(tokens[i])
            merged_indices.append(current_index)

    return merged_tokens

class Evaluator:
    def __init__(self, model, tokenizer, model_handle):
        """
        Initialize the Evaluator class with a tokenizer and model.

        :param tokenizer: The tokenizer to be used for encoding/decoding text.
        :param model: The model to be used for text generation.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.model_handle = model_handle
        # if self.tokenizer.pad_token_id is None:
        #     self.pad_text = self.tokenizer.decode(self.tokenizer.eos_token_id)
        # else:
        if "openai" in str(type(model)):
            self.pad_text = '<|endoftext|>'
        else:
            self.pad_text = self.tokenizer.decode(self.tokenizer.pad_token_id)

    def decode_tokens(self, sequence):

        return [self.tokenizer.decode([token]) for token in sequence]

    def create_masked_sequence(self, eval_sequence, raw_sequence):
        """
        Create a masked sequence based on the evaluation sequence.

        :param eval_sequence: The evaluation sequence of token IDs.
        :param raw_sequence: The raw sequence of token IDs.
        :return: Masked text with placeholders.
        """
        occurrence_indices = find_subsequence_occurrences(eval_sequence, raw_sequence)
        occurrence_ind = occurrence_indices[0]

        masked_sequence = [self.tokenizer.pad_token_id for _ in range(len(raw_sequence))]
        for i, ind in enumerate(occurrence_ind):
            masked_sequence[ind] = eval_sequence[i]

        padded_text_list = self.decode_tokens(masked_sequence)
        masked_text = "".join(padded_text_list).replace(self.pad_text, "<mask>")
        return masked_text

    # def complete_text(self, masked_text, test_sentence_num):
    #     """
    #     Complete the masked text using a BERT model.
    #
    #     :param masked_text: The text with masked tokens.
    #     :param model_name: Name of the BERT model to use.
    #     :param test_sentence_num: Number of sentences to generate.
    #     :return: List of completed sentences.
    #     """
    #     return complete_masked_sentence_bert(masked_text, num_sentences=test_sentence_num)

    def generate_outputs(self, wrapped_texts, max_new_tokens=128):
        """
        Generate predictions based on wrapped filled texts.

        :param wrapped_texts: List of texts to generate predictions for.
        :param max_length: Maximum length for generation.
        :return: Generated token IDs.
        """
        inputs = self.tokenizer(wrapped_texts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
        return outputs.sequences[:, inputs.input_ids.shape[1]:]

    def evaluate(self, eval_sequence, raw_sequence, yh, test_sentence_num=20, device=0, max_new_tokens=128, completion_mode="bert", wrapping_func=lambda x:x, return_texts=True):
        """
        Evaluate the model's ability to complete masked text.

        :param eval_sequence: The evaluation sequence of token IDs.
        :param raw_sequence: The raw sequence of token IDs.
        :param model_name: Name of the BERT model to use.
        :param test_sentence_num: Number of sentences to generate.
        :param yh: Target string to check for in the generated outputs.
        :return: Proportion of outputs containing the target string.
        """
        if completion_mode == "bert":
            masked_text = self.create_masked_sequence(eval_sequence, raw_sequence)
            filled_texts = complete_masked_sentence_bert(masked_text, num_sentences=test_sentence_num, device_index=device)
        elif completion_mode == "openai-token":
            tokens = self.decode_tokens(eval_sequence)
            occurrence_ind = find_subsequence_occurrences(eval_sequence, raw_sequence)[0]
            merged_tokens = merge_consecutive_tokens(tokens, occurrence_ind)
            # filled_texts = complete_tokens_with_chatgpt(merged_tokens, num_sentences=test_sentence_num)
            filled_texts = complete_tokens_with_chatgpt_batch(merged_tokens)
        elif completion_mode == "openai-mask":
            masked_text = self.create_masked_sequence(eval_sequence, raw_sequence)
            # filled_texts = complete_masked_sentence_chatgpt(masked_text, num_sentences=test_sentence_num)
            filled_texts = complete_masked_sentence_chatgpt_batch(masked_text)
        elif completion_mode == "random":
            filled_texts = self.fill_random_tokens(eval_sequence, raw_sequence, num_sentences=test_sentence_num)

        wrapped_filled_texts = [wrapping_func(text) for text in filled_texts]
        if "gpt" in self.model_handle:

            org_len = len(self.tokenizer.encode(wrapped_filled_texts[0]))
            output_texts = generate_chatgpt_batch(wrapped_filled_texts, client=self.model, model_name=self.model_handle,
                                               max_tokens=max_new_tokens + org_len, workers=25)
            count = sum([1 for output_text in output_texts if has_subseq([yh], output_text)])
        else:
            generated_ids = self.generate_outputs(wrapped_filled_texts, max_new_tokens=max_new_tokens)
            count = 0
            output_texts = []
            for i in range(test_sentence_num):
                generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[i])
                cleaned_tokens = [clean_token(token) for token in generated_tokens]
                output_text = "".join(cleaned_tokens)
                output_texts.append(output_text)
                if has_subseq([yh], output_text):
                    count += 1
        if return_texts:
            return count / test_sentence_num, list(zip(filled_texts, output_texts))
        return count / test_sentence_num

    def fill_random_tokens(self, eval_sequence, raw_sequence, num_sentences=10):
        occurrence_indices = find_subsequence_occurrences(eval_sequence, raw_sequence)
        occurrence_ind = occurrence_indices[0]

        filled_texts = []
        for i in range(num_sentences):

            masked_sequence = [random.randint(100, self.tokenizer.vocab_size - 100) for _ in range(len(raw_sequence))]
            for i, ind in enumerate(occurrence_ind):
                masked_sequence[ind] = eval_sequence[i]

            filled_text = "".join(self.decode_tokens(masked_sequence))
            filled_texts.append(filled_text)

        return filled_texts

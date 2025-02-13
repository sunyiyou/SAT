
import re
import random
from transformers import pipeline

def complete_masked_sentence_bert(incomplete_sentence, model_name="FacebookAI/roberta-base", num_sentences=5, device_index=0):
    """
    Completes a sentence with <mask> tokens using a fill-mask model,
    generating multiple variations by filling <mask> tokens at random positions.

    Args:
        incomplete_sentence (str): The sentence containing <mask> tokens to be filled.
        model_name (str): The name of the model to use (e.g., 'bert-base-cased').
        num_sentences (int): The number of completed sentences to generate.

    Returns:
        List[str]: A list of completed sentences.
    """
    # Initialize the fill-mask pipeline
    fill_mask = pipeline("fill-mask", model=model_name, device=device_index)

    # Store the generated sentences
    completed_sentences = []

    for _ in range(num_sentences):
        # Start with the original text for each generated sentence
        text = incomplete_sentence

        # Repeat until no more <mask> tokens remain in the text
        while "<mask>" in text:
            # Get predictions from the pipeline
            suggestions = fill_mask(text)

            # ─────────────────────────────────────────────────────────────────
            #  DETECT WHETHER WE HAVE A SINGLE OR MULTIPLE MASKS
            # ─────────────────────────────────────────────────────────────────

            if not suggestions:
                # Edge case: If for some reason we get an empty result, break to avoid infinite loop
                break

            # If suggestions[0] is a dict, there's only ONE mask in the text
            # If suggestions[0] is a list, there are MULTIPLE masks in the text
            single_mask = isinstance(suggestions[0], dict)

            if single_mask:
                # We only have one <mask> in the entire text
                best_suggestion = suggestions[0]["token_str"]

                # Optionally: Add randomness in selecting among top suggestions
                # best_suggestion = random.choice(suggestions[:5])["token_str"]

                # Replace the single <mask> in text
                text = text.replace("<mask>", best_suggestion, 1)

            else:
                # We have multiple <mask> tokens in the text
                # Randomly pick which <mask> to fill this time
                i = random.randrange(len(suggestions))

                # Pick the top suggestion for that i-th mask
                best_suggestion = suggestions[i][0]["token_str"]

                # Optionally: Add randomness in selecting among top suggestions
                # best_suggestion = random.choice(suggestions[i][:5])["token_str"]

                # Find the i-th occurrence of <mask> in text (left-to-right)
                mask_positions = [m.start() for m in re.finditer("<mask>", text)]
                mask_pos = mask_positions[i]

                # Replace only this i-th occurrence of <mask>
                text = text[:mask_pos] + best_suggestion + text[mask_pos+len("<mask>"):]

        # Append the completed sentence to the list
        completed_sentences.append(text)

    return completed_sentences

if __name__ == '__main__':

    # Example usage:

    model_name = "FacebookAI/roberta-base"

    incomplete_sentence = (
        '<mask><mask> in<mask><mask><mask><mask><mask><mask> singer<mask> fame<mask><mask><mask><mask><mask><mask><mask> tragedy<mask><mask><mask><mask><mask><mask><mask><mask>Elvis<mask><mask>?'
    )
    completed_sentences = complete_masked_sentence_bert(incomplete_sentence, model_name=model_name, num_sentences=3)
    print(completed_sentences)



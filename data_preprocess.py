import os
import re
from utils.general import *
from utils.causal_analyzer import SubsequenceCausalAnalyzer
import pandas as pd
import argparse
import ast
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Script to configure and run a model with specific parameters.")
    parser.add_argument("--model_name", type=str, default="llama_3_70b_chat", help="OLMo-7B-Instruct, llama_3_70b_chat")
    parser.add_argument("--halo_type", type=str, default="rationalization_binary_prime", help="Type of the halo to use (e.g., 'code', 'biographies', 'numerical_falsepresupposition', 'rationalization_binary_prime', 'rationalization_binary_senator', 'rationalization_numerical', 'references').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    halo_type = args.halo_type
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens

    output_folder = f"halu_results/{model_name}/{halo_type}"
    os.makedirs(output_folder, exist_ok=True)
    response_file = os.path.join(output_folder, f"{halo_type}_responses.csv")
    response_halu_file = os.path.join(output_folder, f"{halo_type}_responses_halu.csv")

    if not os.path.exists(response_halu_file):

        if args.halo_type == "code":
            halo_path = f"HALoGEN/model_hallucinations/code/{model_name.lower().replace('-','_')}_code_AU.csv"
            df = pd.read_csv(halo_path)
            def remove_element(lst, element_to_remove):
                return [x for x in lst if x not in element_to_remove]

            prefix_context = "Write a Python program that utilizes modules, packages, and/or libraries to solve the problem: "
            # Check the result
            print(df["hallucinated_atomic_units"].head())
            df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(ast.literal_eval)
            df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(lambda x: remove_element(x, ('cv2', 'cProfile', 'googleapiclient', 'git', 'googlesearch', 'speech_recognition', 'rest_framework', 'dateutil', 'paho', 'RPi', 'gi', 'pkg_resources')))
            filtered_df = df[
                (df["hallucinated_atomic_units"].apply(len) > 0) & (df["prompt"].str.startswith(prefix_context))
            ]
            filtered_df.to_csv(os.path.join(output_folder, f"{args.halo_type}_responses_halu.csv"), index=True)

        if args.halo_type == "biographies":
            halo_path = f"HALoGEN/model_hallucinations/biographies/{model_name.lower().replace('-','_')}_biographies.csv"
            df = pd.read_csv(halo_path)
            place_extract = lambda input_text: re.findall(r"was born in ([^.]+)\.", str(input_text))
            df["hallucinated_atomic_units"] = df["hallucinated atoms"].apply(place_extract)
            filtered_df = df[df["hallucinated_atomic_units"].apply(len) > 0].copy()
            filtered_df["hallucinated_atomic_units"] = filtered_df["hallucinated_atomic_units"].apply(lambda l: [('born', item) for item in l])
            filtered_df.to_csv(os.path.join(output_folder, f"{args.halo_type}_responses_halu.csv"), index=True)

        if args.halo_type == "numerical_falsepresupposition":
            halo_path = f"HALoGEN/model_hallucinations/{args.halo_type}/{model_name.lower().replace('-','_')}_numerical_inconsistency_AU.csv"
            df = pd.read_csv(halo_path)

            suffix_context = " Separate the items of the list using semicolons as separators. Provide only the list without any additional information. If you cannot answer, respond with 'no response.'"

            def safe_eval_and_filter(x):
                if isinstance(x, str) and x.strip():
                    try:
                        return [item for item in ast.literal_eval(x) if isinstance(item, str) and len(item) > 3]
                    except (ValueError, SyntaxError):
                        return []
                else:
                    return []

            df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(safe_eval_and_filter)
            filtered_df = df[
                (df["hallucinated_atomic_units"].apply(len) > 0) &
                (df["prompt"].str.endswith(suffix_context)) &
                df.apply(lambda row: any(item in row["prompt"] for item in row["hallucinated_atomic_units"]), axis=1)
            ]
            filtered_df.to_csv(os.path.join(output_folder, f"{args.halo_type}_responses_halu.csv"), index=True)

        if args.halo_type == "rationalization_binary_prime":
            halo_path = f"HALoGEN/model_hallucinations/rationalization_binary/{model_name.lower().replace('-','_')}_yesno_response_AU.csv"
            df = pd.read_csv(halo_path)
            suffix_context = " First, respond with yes or no. If no, then provide its factorization."

            def safe_eval_and_filter(x):
                if isinstance(x, str) and x.strip():
                    try:
                        num_set = []
                        for num in set([item for item in ast.literal_eval(x) if item not in {'yes', 'no'}]):
                            if "olmo" in model_name.lower():
                                num_set.extend([f"x {num}", f"× {num}", f"* {num}", f"x{num}", f"× {num}", f"*{num}", f"{num} x", f"{num} ×", f"{num} *", f"{num}x", f"{num}×", f"{num}*", f"{num}^"])
                            else:
                                num_set.extend([f" {num}", f"{num} "])
                        return num_set
                    except (ValueError, SyntaxError):
                        return []
                else:
                    return []

            df_prime = df[df["prompt"].str.contains("prime", na=False)].copy()
            df = df_prime
            df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(safe_eval_and_filter)
            filtered_df = df[ (df["hallucinated_atomic_units"].apply(len) > 0) & (df["prompt"].str.endswith(suffix_context))]
            filtered_df.to_csv(os.path.join(output_folder, f"{args.halo_type}_responses_halu.csv"), index=True)


        if args.halo_type == "rationalization_binary_senator":
            halo_path = f"HALoGEN/model_hallucinations/rationalization_binary/{model_name.lower().replace('-', '_')}_yesno_response_AU.csv"
            df = pd.read_csv(halo_path)
            suffix_context = " First, respond with yes or no. If yes, then provide the name of the US senator."

            def safe_eval_and_filter(x):
                if isinstance(x, str) and x.strip():
                    try:
                        return [item for item in ast.literal_eval(x) if isinstance(item, str) and item not in {'yes', 'no'}]
                    except (ValueError, SyntaxError):
                        return []
                else:
                    return []

            df_senator = df[df["prompt"].str.contains("US senator", na=False)].copy()
            df = df_senator
            df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(safe_eval_and_filter)
            filtered_df = df[
                (df["hallucinated_atomic_units"].apply(len) > 0) & (df["prompt"].str.endswith(suffix_context))
            ]
            filtered_df.to_csv(os.path.join(output_folder, f"{args.halo_type}_responses_halu.csv"), index=True)

        if args.halo_type == "rationalization_numerical":
            halo_path = f"HALoGEN/model_hallucinations/rationalization_numerical/{model_name.lower().replace('-', '_')}_numerical_response_AU.csv"
            df = pd.read_csv(halo_path)
            suffix_context = " First output a number, and then list every item that satisfies the condition."

            def safe_eval_and_filter(x):
                if isinstance(x, str) and x.strip():
                    try:
                        return [item for item in ast.literal_eval(x) if len(str(item)) > 3]
                    except (ValueError, SyntaxError):
                        return []
                else:
                    return []

            df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(safe_eval_and_filter)
            filtered_df = df[
                (df["hallucinated_atomic_units"].apply(len) > 0) & (df["prompt"].str.endswith(suffix_context))
            ]
            filtered_df.to_csv(os.path.join(output_folder, f"{args.halo_type}_responses_halu.csv"), index=True)

        if args.halo_type == "references":
            halo_path = f"HALoGEN/model_hallucinations/references/{model_name.lower().replace('-', '_')}_references.csv"
            df = pd.read_csv(halo_path)

            prefix_context = " Find relevant scientific or academic references supporting the following Question-Answer pair in APA format. Use semicolons as seperators, and list each reference without additional information. "

            def safe_eval_and_filter(x):
                if isinstance(x, str) and x.strip():
                    try:
                        return [item for item in ast.literal_eval(x) if len(str(item)) > 3 and len(str(item)) < 50]
                    except (ValueError, SyntaxError):
                        return []
                else:
                    return []

            df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(safe_eval_and_filter)
            filtered_df = df[
                (df["hallucinated_atomic_units"].apply(len) > 0) & (df["prompt"].str.startswith(prefix_context))
            ]
            filtered_df.to_csv(os.path.join(output_folder, f"{args.halo_type}_responses_halu.csv"), index=True)

    else:
        df = pd.read_csv(response_halu_file)
        df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(ast.literal_eval)


    if "olmo" in model_name.lower():
        chat_prefix_default = "<|endoftext|><|user|>\n"
        chat_suffix_default = "\n<|assistant|>\n"
    elif "llama" in model_name.lower():
        chat_prefix_default = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        chat_suffix_default = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    else:
        chat_prefix_default = ""
        chat_suffix_default = ""
    apply_chat_template_default = lambda pr: f"{chat_prefix_default}{pr}{chat_suffix_default}"

    output_folder = f"halu_results/{model_name}/{halo_type}"
    response_halu_freqs_file = os.path.join(output_folder, f"{halo_type}_responses_halu_freqs.csv")
    print("# Step 1: make generation and calculate initial frequence")
    if not os.path.exists(response_halu_freqs_file):
        model_handle = MODEL_NAME_DICT[model_name]
        response_halu_file = os.path.join(output_folder, f"{halo_type}_responses_halu.csv")
        df = pd.read_csv(response_halu_file)
        df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(ast.literal_eval)
        prompt_inputs = df["prompt"].tolist()
        response_outputs = df["response"].tolist()
        triggers = df["hallucinated_atomic_units"].tolist()
        analyzer = SubsequenceCausalAnalyzer(model_handle=model_handle, chat_suffix=chat_suffix_default, chat_prefix=chat_prefix_default)

        freqs = []
        for index, (prompt_text, response, target_string) in tqdm(enumerate(zip(prompt_inputs, response_outputs, triggers)), total=len(prompt_inputs)):
            # Step 1: Prompt and generation
            prompt_text = prompt_text.replace(chat_prefix_default, "").replace(chat_suffix_default, "")
            gen_results = analyzer.generate_response(prompt_text, max_new_tokens=max_new_tokens, num_generations=batch_size)
            indices = [i for i, gen_result in enumerate(gen_results) if has_subseq(target_string, gen_result["output_text"])]
            pyh = len(indices) / len(gen_results)
            print(f"Target '{target_string}' appeared in {len(indices)} out of {len(gen_results)} outputs. p(yh|s) = {pyh:.2f}")
            freqs.append(pyh)

        df['freqs'] = freqs
        df = df.sort_values(by='freqs', ascending=False)
        df.to_csv(response_halu_freqs_file, index=False)

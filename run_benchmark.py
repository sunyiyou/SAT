import math
import pickle
from utils.causal_analyzer import SubsequenceCausalAnalyzer
from utils.general import MODEL_NAME_DICT, ROOT, clean_token, has_subseq
from utils.html_report import HTMLReporter
import datetime, os
import argparse
import pandas as pd
from tqdm import tqdm
import ast
import torch
import csv
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Script to configure and run a model with specific parameters.")
    parser.add_argument("--model_name", type=str, default="llama_3_70b_chat", help="model name. llama_3_70b_chat, OLMo-7B-Instruct")
    parser.add_argument("--halo_type", type=str, default="code", help="Type of the halo to use (e.g., 'code', 'biographies', 'numerical_falsepresupposition', 'rationalization_binary_senator', 'rationalization_binary_prime', 'rationalization_numerical', 'references').")
    parser.add_argument("--generation_mode", type=str, default="chat", help="Mode of generation (e.g., 'chat').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to sample during generation.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use.")

    parser.add_argument("--perturbation_rate", type=float, default=0.5, help="Rate of perturbation to be applied (default: 0.5)")
    parser.add_argument("--max_subseq_len_rate", type=int, default=0.9, help="Length of the subsequence to be used (default: 12)")
    parser.add_argument("--min_level", type=int, default=4, help="Minimum level of perturbation (default: 4)")
    parser.add_argument("--search_beam", type=int, default=20, help="Minimum level of perturbation (default: 4)")
    parser.add_argument("--test_sentence_num", type=int, default=25, help="Number of sentences to test (default: 20)")
    parser.add_argument("--completion_modes", type=str, nargs="+", default=['random', 'bert',  "openai-mask", "openai-token"], help="Modes of completion to be tested (default: ['bert']) ['bert', 'random', 'openai-mask', 'openai-token',]")
    parser.add_argument("--num_perturbations", type=int, default=1024, help="Number of perturbations to generate (default: 500)")
    parser.add_argument("--perturbation_mode", type=str, default="rgrand", choices=["rgrand", "rand"], help="Perturbation mode to use, 'rgrand' or 'rand' (default: 'rgrand')")
    parser.add_argument("--eval_wrap_func_mode", type=str, default="chat", choices=["chat", "none"], help="Evaluation wrapper function mode, 'chat' or 'none' (default: 'chat')")
    parser.add_argument("--eval_level_range", type=str, default="quad", help="")
    parser.add_argument("--eval_samples_max", type=int, default=100, help="")

    parser.add_argument("--do_context", type=bool, default=True, help="Whether to sample during generation.")
    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':

    model_name = args.model_name
    model_handle = MODEL_NAME_DICT[model_name]
    halo_type = args.halo_type
    generation_mode = args.generation_mode
    batch_size = args.batch_size
    do_sample = args.do_sample
    do_context = args.do_context
    beam = args.search_beam
    max_new_tokens = args.max_new_tokens
    perturbation_rate = args.perturbation_rate
    max_subseq_len_rate = args.max_subseq_len_rate
    min_level = args.min_level
    test_sentence_num = args.test_sentence_num
    completion_modes = args.completion_modes
    num_perturbations = args.num_perturbations
    perturbation_mode = args.perturbation_mode
    eval_wrap_func_mode = args.eval_wrap_func_mode
    eval_level_range = args.eval_level_range
    eval_samples_max = args.eval_samples_max

    if "70b" not in model_name:
        torch.cuda.set_device(args.gpu_id)

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
    if args.do_context:
        tag = f"setting-@pert-num{num_perturbations}-rate{perturbation_rate}-beam{beam}-mode{perturbation_mode}-maxsqlr{max_subseq_len_rate}"
    else:
        tag = f"setting-nocontext-@pert-num{num_perturbations}-rate{perturbation_rate}-beam{beam}-mode{perturbation_mode}-maxsqlr{max_subseq_len_rate}"

    save_folder = os.path.join(ROOT, output_folder, tag)
    os.makedirs(save_folder, exist_ok=True)

    response_halu_freqs_file = os.path.join(output_folder, f"{halo_type}_responses_halu_freqs.csv")
    df = pd.read_csv(response_halu_freqs_file, index_col=0)
    df["hallucinated_atomic_units"] = df["hallucinated_atomic_units"].apply(ast.literal_eval)

    analyzer = SubsequenceCausalAnalyzer(model_handle=model_handle, chat_suffix=chat_suffix_default, chat_prefix=chat_prefix_default)

    df_eval_scores_all = []
    cnt = 0
    # Step 2: Analyze subsequences (perturbations, frequency of target string)
    for index in tqdm(df.index):
        prompt_text = df.loc[index]["prompt"]
        response = df.loc[index]["response"]
        freq = df.loc[index]["freqs"]
        target_string = df.loc[index]["hallucinated_atomic_units"]
        if df.loc[index]["freqs"] < 0.1: continue
        cnt +=1
        if cnt > eval_samples_max: break
        if not has_subseq([target_string], response): break
        analysis_result_file = os.path.join(save_folder, f"{index}_analysis_result.pickle")
        perturb_result_file = os.path.join(save_folder, f"{index}_perturb_result.csv")

        if args.do_context:
            prefix_context = "*!@#^%!@*#^%!@*#^&!%@$" # nonsense string
            suffix_context = "*!@#^%!@*#^%!@*#^&!%@$" # nonsense string
            if halo_type == "code":
                prefix_context = "Write a Python program that utilizes modules, packages, and/or libraries to solve the problem: "
                chat_prefix = chat_prefix_default + prefix_context
                chat_suffix = chat_suffix_default
            elif halo_type == "biographies":
                chat_prefix = chat_prefix_default
                chat_suffix = chat_suffix_default
            elif halo_type == "numerical_falsepresupposition":
                chat_prefix = chat_prefix_default
                suffix_context = " Separate the items of the list using semicolons as separators. Provide only the list without any additional information. If you cannot answer, respond with 'no response.'"
                chat_suffix = suffix_context+chat_suffix_default
            elif halo_type == "rationalization_binary_prime":
                chat_prefix = chat_prefix_default
                suffix_context = " First, respond with yes or no. If no, then provide its factorization."
                chat_suffix = suffix_context+chat_suffix_default
            elif halo_type == "rationalization_binary_senator":
                chat_prefix = chat_prefix_default
                suffix_context = " First, respond with yes or no. If yes, then provide the name of the US senator."
                chat_suffix = suffix_context+chat_suffix_default
            elif halo_type == "rationalization_numerical":
                chat_prefix = chat_prefix_default
                suffix_context = " First output a number, and then list every item that satisfies the condition."
                chat_suffix = suffix_context+chat_suffix_default
            elif halo_type == "references":
                prefix_context = " Find relevant scientific or academic references supporting the following Question-Answer pair in APA format. Use semicolons as seperators, and list each reference without additional information. "
                chat_prefix = chat_prefix_default + prefix_context
                chat_suffix = chat_suffix_default
            else:
                1/0

            prompt_text_dewrap = prompt_text.replace(chat_suffix_default, "").replace(suffix_context, "").replace(chat_prefix_default, "").replace(prefix_context, "")
            analyzer.chat_prefix = chat_prefix
            analyzer.chat_suffix = chat_suffix
        else:
            prompt_text_dewrap = prompt_text.replace(chat_suffix_default, "").replace(chat_prefix_default, "")
            analyzer.chat_prefix = chat_prefix_default
            analyzer.chat_suffix = chat_suffix_default

        if not os.path.exists(analysis_result_file) or not os.path.exists(perturb_result_file):
            analysis_results = analyzer.analyze_subsequences(
                prompt=prompt_text_dewrap,
                target_string=target_string,
                num_perturbations=num_perturbations,
                perturbation_rate=perturbation_rate,
                do_sample=do_sample,
                perturbation_mode=perturbation_mode,
                max_subseq_len_rate=max_subseq_len_rate,
                ignore_items=set(),
                beam = beam,
            )

            with open(analysis_result_file, 'wb') as f:
                pickle.dump(analyzer.analysis_results, f)

            with open(perturb_result_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(("perturbed_input", "output"))
                for row in analyzer.perturbed_results:
                    writer.writerow(row)
        else:
            df_pert_result = pd.read_csv(perturb_result_file)
            analyzer.perturbed_results = list(df_pert_result.itertuples(index=False, name=None))
            with open(analysis_result_file, 'rb') as f:
                analyzer.analysis_results = pickle.load(f)

        # Step 3: Generate evaluation summaries
        # completion_modes = ["bert", "random", "openai-mask", "openai-token"]
        eval_summaries_all = {}
        eval_save_folder = os.path.join(save_folder, f"eval-@test_num{test_sentence_num}-mode{eval_wrap_func_mode}-range{eval_level_range}")
        for completion_mode in completion_modes:
            eval_save_mode_folder = os.path.join(eval_save_folder, f"{completion_mode}")
            os.makedirs(eval_save_mode_folder, exist_ok=True)
            eval_result_file = os.path.join(eval_save_mode_folder, f"{index}_eval.pickle")
            if not os.path.exists(eval_result_file):
                if eval_level_range == "all":
                    level_range = list(range(min_level, len(analyzer.analysis_results["freq_sequence_at_level"])+1))
                elif eval_level_range == "mid":
                    level_range = [math.ceil(len(analyzer.analysis_results["freq_sequence_at_level"]) / max_subseq_len_rate * 0.5)]
                elif eval_level_range == "quad":
                    level_range = [math.ceil(len(analyzer.analysis_results["freq_sequence_at_level"]) / max_subseq_len_rate * 0.25),
                                   math.ceil(len(analyzer.analysis_results["freq_sequence_at_level"]) / max_subseq_len_rate * 0.5),
                                   math.ceil(len(analyzer.analysis_results["freq_sequence_at_level"]) / max_subseq_len_rate * 0.75)]

                eval_summaries = analyzer.generate_eval_summaries(
                    prompt=prompt_text_dewrap,
                    target_string=target_string,
                    test_sentence_num=test_sentence_num,
                    completion_mode=completion_mode,
                    device=int(args.gpu_id),
                    level_range=level_range,
                    wrapping_func={"chat": analyzer.apply_chat_template, "none": lambda x: x}[eval_wrap_func_mode],
                )
                with open(eval_result_file, 'wb') as f:
                    pickle.dump(eval_summaries, f)
            else:
                with open(eval_result_file, 'rb') as f:
                    eval_summaries = pickle.load(f)
            eval_summaries_all[completion_mode] = eval_summaries

        # merge them
        eval_summaries_merged = copy.deepcopy(eval_summaries)
        for level, content in eval_summaries_merged.items():
            del eval_summaries_merged[level]["evaluation"]
            eval_summaries_merged[level]["evaluation"] = {}
        for completion_mode in completion_modes:
            eval_summaries_mode = eval_summaries_all[completion_mode]
            for level, content in eval_summaries_mode.items():
                mode = content["evaluation"]["mode"]
                eval_summaries_merged[level]["evaluation"][mode] = {
                    "test_rate": content["evaluation"]["test_rate"],
                    "samples": content["evaluation"]["eval_texts"]
                }
        #%%
        html_file = os.path.join(eval_save_folder, f"{index}.html")
        if not os.path.exists(html_file):
            subseqs_at_levels = {}
            for lvl, (best_subseq_tokens, score) in analyzer.analysis_results["freq_sequence_at_level"].items():
                if "gpt" in model_handle:
                    substr_tokens = [analyzer.tokenizer.decode([t]) for t in best_subseq_tokens]
                else:
                    substr_tokens = [clean_token(analyzer.tokenizer.convert_ids_to_tokens(t)) for t in best_subseq_tokens]
                subseqs_at_levels[lvl] = (substr_tokens, score)
            # Step 4: Create an instance of HTMLReporter
            reporter = HTMLReporter(
                model_name=model_name,
                num_perturbations=num_perturbations,
                perturbation_rate=perturbation_rate,
                subsequence_length=len(analyzer.analysis_results["freq_sequence_at_level"]),
                prompt_text=prompt_text,
                output_text=response,
                target_string=target_string,
                perturbed_samples=analyzer.perturbed_results,
                subseqs_at_levels=subseqs_at_levels,
                eval_summaries=eval_summaries_merged
            )
            # Generate the HTML content
            html_report = reporter.generate_html_report()
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_report)

        # Step 5: Summarize all evaluation results
        rows = []
        for lvl, data in sorted(eval_summaries_merged.items(), key=lambda x: x[0]):
            best_score = data.get("best_score", -1)
            completion_results = data.get("evaluation", {})
            test_rates = []
            for mode in completion_modes:
                mode_data = completion_results.get(mode, {})
                test_rate = mode_data.get("test_rate", -1)
                test_rates.append(test_rate)
            avg_test_rate = sum(test_rates) / len(test_rates) if test_rates else 0.0
            row = {
                "Index": index,
                "Level": lvl,
                "Best Score": best_score,
                "Average Test Rate": avg_test_rate
            }
            for mode in completion_modes:
                mode_data = completion_results.get(mode, {})
                test_rate = mode_data.get("test_rate", -1)
                row[f"{mode} Test Rate"] = test_rate
            rows.append(row)
        # Convert to DataFrame
        df_eval_score = pd.DataFrame(rows)
        df_eval_scores_all.append(df_eval_score)

    df_eval_scores_all = pd.concat(df_eval_scores_all, ignore_index=True)
    df_eval_scores_all.to_csv(os.path.join(eval_save_folder, "all_scores.csv"), index=False)



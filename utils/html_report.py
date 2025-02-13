

class HTMLReporter:
    """
    A class to hold analysis results and produce an HTML report for:
      1. Basic settings
      2. Example perturbed sequences
      3. Subsequences found at different levels
      4. Evaluation results for different completion modes
    """

    def __init__(
            self,
            model_name: str,
            num_perturbations: int,
            perturbation_rate: float,
            subsequence_length: int,
            prompt_text: str,
            output_text: str,
            target_string: str,
            perturbed_samples: list,
            subseqs_at_levels: dict,
            eval_summaries: dict
    ):
        """
        Store all analysis data in the class instance.

        :param model_name: Name of the loaded model.
        :param num_perturbations: Total number of perturbed sequences.
        :param perturbation_rate: Fraction of tokens replaced by pad tokens.
        :param subsequence_length: Max length for subsequence search.
        :param prompt_text: The original prompt used.
        :param target_string: Target string being checked in the output (e.g., "Presley").
        :param perturbed_samples: A few examples of perturbed sequences in string format.
        :param subseqs_at_levels: Dict of the form:
            {
              level: (best_subseq_tokens, freq_score, test_rate, eval_texts_by_mode),
              ...
            }
        :param eval_summaries: Detailed evaluation results for each level and each completion mode.
            For instance:
            {
              level: {
                "best_subseq": [...],
                "evaluation": {
                  "bert": {
                    "test_rate": float,
                    "samples": [("input", "output"), ...]
                  },
                  "random": {...},
                  ...
                }
              }
            }
        """
        self.model_name = model_name
        self.num_perturbations = num_perturbations
        self.perturbation_rate = perturbation_rate
        self.subsequence_length = subsequence_length
        self.prompt_text = prompt_text
        self.output_text = output_text
        self.target_string = target_string
        self.perturbed_samples = perturbed_samples
        self.subseqs_at_levels = subseqs_at_levels
        self.eval_summaries = eval_summaries

    def _make_io_table(self, io_pairs):
        """
        Helper method to create an HTML table for input-output pairs,
        limited to 10 rows for readability.

        :param io_pairs: list of tuples: (input_text, output_text)
        :return: str (HTML)
        """
        max_examples = min(len(io_pairs), 10)
        table_html = []
        table_html.append("<table>")
        table_html.append("<tr><th style='width:50%'>Input</th><th style='width:50%'>Output</th></tr>")

        for i in range(max_examples):
            # io_id, (inp, outp) = io_pairs[i]
            inp, outp = io_pairs[i]
            # Minimal HTML escaping or replacing if needed
            table_html.append(f"<tr><td><pre>{inp}</pre></td><td><pre>{outp}</pre></td></tr>")

        table_html.append("</table>")
        return "\n".join(table_html)

    def create_test_rate_table(self):
        """
        Generates an HTML table that summarizes:
            - the best_score for each level,
            - the average (mean) test_rate across all modes for that level, and
            - the individual test_rate values for each mode.
        """

        # 1) Gather all unique completion modes across levels
        all_modes = set()
        for lvl, data in self.eval_summaries.items():
            completion_results = data.get("evaluation", {})
            for mode_name in completion_results.keys():
                all_modes.add(mode_name)

        # Sort modes for a consistent column order
        all_modes = sorted(all_modes)

        # 2) Build the HTML table (head + body)
        table_html = []
        table_html.append("<table border='1' style='border-collapse: collapse;'>")
        table_html.append("<thead>")
        table_html.append("<tr>")
        table_html.append("<th>Level</th>")
        table_html.append("<th>Best Score</th>")
        table_html.append("<th>Avg. Test Rate</th>")  # New column
        for mode in all_modes:
            table_html.append(f"<th>{mode} (Test Rate)</th>")
        table_html.append("</tr>")
        table_html.append("</thead>")

        table_html.append("<tbody>")
        # Sort levels numerically (or however you prefer)
        for lvl, data in sorted(self.eval_summaries.items(), key=lambda x: x[0]):

            # Extract best_score
            best_score = data.get("best_score", -1)
            # Extract completion results for modes
            completion_results = data.get("evaluation", {})

            # Collect test rates from all modes for averaging
            test_rates = []
            for mode in all_modes:
                mode_data = completion_results.get(mode, {})
                test_rate = mode_data.get("test_rate", -1)
                test_rates.append(test_rate)

            # Compute average test rate (avoid division by zero)
            if test_rates:
                avg_test_rate = sum(test_rates) / len(test_rates)
            else:
                avg_test_rate = 0.0

            # Start the row: [Level, Best Score, Avg. Test Rate]
            row_html = [
                f"<td>{lvl}</td>",
                f"<td>{best_score:.3f}</td>",
                f"<td>{avg_test_rate:.3f}</td>"
            ]

            # Add one cell per mode's test_rate
            for mode in all_modes:
                mode_data = completion_results.get(mode, {})
                test_rate = mode_data.get("test_rate", -1)
                row_html.append(f"<td>{test_rate:.3f}</td>")

            table_html.append(f"<tr>{''.join(row_html)}</tr>")
        table_html.append("</tbody>")
        table_html.append("</table>")

        # 3) Return the concatenated HTML string
        return "\n".join(table_html)

    def generate_html_report(self):
        """
        Create and return a single HTML string that includes:
          1. Basic settings
          2. 5 examples of perturbed sequences
          3. Subsequences found at different levels
          4. Evaluation results across multiple completion modes (bert, random, openai-mask, openai-token)

        :return: str, HTML content
        """
        html_content = []

        # HTML Header
        html_content.append("<html>")
        html_content.append("<head>")
        html_content.append("<meta charset='UTF-8' />")
        html_content.append("<title>Subsequence Analysis Report</title>")
        html_content.append("""
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            h2, h3, h4 {
                margin-top: 20px;
                color: #2c3e50;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }
            .eval-block {
                margin-bottom: 20px;
                padding: 10px;
                background-color: #f4f4f4;
                border-radius: 5px;
            }
            .level-block {
                margin-bottom: 20px;
            }
            pre {
                margin: 0; 
                white-space: pre-wrap; 
                word-wrap: break-word;
            }
        </style>
        """)
        html_content.append("</head>")

        # Body starts
        html_content.append("<body>")
        html_content.append("<h2>Subsequence Analysis Report</h2>")

        # (1) Basic Settings
        html_content.append("<h3>1. Basic Settings</h3>")
        html_content.append("<ul>")
        html_content.append(f"<li><strong>Model Name:</strong> {self.model_name}</li>")
        html_content.append(f"<li><strong>Num Perturbations:</strong> {self.num_perturbations}</li>")
        html_content.append(f"<li><strong>Perturbation Rate:</strong> {self.perturbation_rate}</li>")
        html_content.append(f"<li><strong>Subsequence Length (max):</strong> {self.subsequence_length}</li>")
        html_content.append("<li><strong>Prompt:</strong> <pre>" + self.prompt_text + "</pre></li>")
        html_content.append("<li><strong>Output:</strong> <pre>" + self.output_text + "</pre></li>")
        html_content.append(f"<li><strong>Target String:</strong> {self.target_string}</li>")
        html_content.append("</ul>")

        # (2) Examples of Perturbed Sequences
        html_content.append("<h3>2. Example Perturbed Sequences</h3>")
        html_content.append("<p>Here are a few representative samples of the perturbed input sequences:</p>")
        # html_content.append("<ol>")
        # max_show = min(len(self.perturbed_samples), 10)
        # for i in range(max_show):
        #     html_content.append(f"<li><pre>{self.perturbed_samples[i]}</pre></li>")
        # html_content.append("</ol>")
        table_html = self._make_io_table(self.perturbed_samples)
        html_content.append(table_html)


        # (3) Subsequences Found at Different Levels
        html_content.append("<h3>3. Subsequences Found at Different Levels</h3>")
        html_content.append(
            "<p>We list the best subsequence discovered at each level along with its frequency score.</p>")
        for lvl, (best_subseq_tokens, score) in self.subseqs_at_levels.items():
            html_content.append(f"<div class='level-block'>")
            html_content.append(f"<h4>Level {lvl}</h4>")
            html_content.append(f"<p><strong>Best Subsequence Tokens:</strong> {best_subseq_tokens}</p>")
            html_content.append(f"<p><strong>p(yh|xs):</strong> {score:.3f}</p>")
            html_content.append("</div>")

        # (4) Evaluation Results
        html_content.append("<h3>4. Evaluation Results</h3>")

        html_content.append("<h4>Test Rate Summary Table</h4>")
        test_rate_table_html = self.create_test_rate_table()  # Use the function
        html_content.append(test_rate_table_html)

        html_content.append(
            "<p>Below we vary the completion mode in [\"bert\", \"random\", \"openai-mask\", \"openai-token\"] and show evaluation results including some example I/O pairs.</p>")

        # We assume eval_summaries is structured like:
        # {
        #   level: {
        #     "best_subseq": [...],
        #     "evaluation": {
        #       "bert": {
        #         "test_rate": float,
        #         "samples": [("input", "output"), ...]
        #       },
        #       "random": {...},
        #       ...
        #     }
        #   }
        # }
        for lvl, data in self.eval_summaries.items():
            best_subseq = data.get("best_subseq", [])
            best_substr = data.get("best_substr", [])
            best_score = data.get("best_score", [])
            completion_results = data.get("evaluation", {})

            html_content.append("<details class='level-block'>")
            html_content.append(f"<summary><strong>Level {lvl}</strong></summary>")

            html_content.append(f"<p><strong>Best Subsequence:</strong> {best_substr}</p>")
            html_content.append(f"<p><strong>p(yh|xs):</strong> {best_score:.3f}</p>")

            for mode_name, mode_data in completion_results.items():
                test_rate = mode_data.get("test_rate", -1)
                samples = mode_data.get("samples", [])
                html_content.append("<div class='eval-block'>")
                html_content.append(f"<h5>Completion Mode: {mode_name}</h5>")
                html_content.append(f"<p><strong>Test Rate:</strong> {test_rate:.3f}</p>")

                # Show table of up to 10 examples
                if samples:
                    table_html = self._make_io_table(samples)
                    html_content.append(table_html)
                else:
                    html_content.append("<p>No samples available.</p>")

                html_content.append("</div>")  # end of eval-block

            html_content.append("</details>")  # end of level-block

        # Close body and html
        html_content.append("</body>")
        html_content.append("</html>")

        # Combine into a single string
        return "\n".join(html_content)


# -----------------------------------------------------------
# Example usage of the HTMLReporter class:
# -----------------------------------------------------------
if __name__ == "__main__":
    # Dummy data for demonstration:
    model_name = "allenai/OLMo-2-1124-7B-Instruct"
    num_perturbations = 100
    perturbation_rate = 0.5
    subsequence_length = 12
    prompt_text = "With roots in New York, this hugely successful singer..."
    target_string = "Presley"

    # A few sample perturbed strings
    perturbed_samples = [
        "With roots in [PAD], this hugely [PAD] singer...",
        "With [PAD] in New York, this hugely successful singer...",
        "[PAD] roots in New York, [PAD] hugely successful singer...",
        "With roots in New [PAD], this hugely [PAD] singer...",
        "With roots in [PAD], this [PAD] successful singer..."
    ]

    # Suppose we found a subsequence at each level (dummy data):
    # { level: (best_subseq_tokens, freq_score, test_rate, eval_texts_by_mode) }
    subseqs_at_levels = {
        3: ([101, 202, 303], 0.45, None, None),
        4: ([101, 202, 303, 404], 0.52, None, None),
        5: ([101, 202, 303, 404, 505], 0.60, None, None),
    }

    # Detailed evaluation data (dummy):
    eval_summaries = {
        3: {
            "best_subseq": [101, 202, 303],
            "evaluation": {
                "bert": {
                    "test_rate": 0.70,
                    "samples": [
                        ("<input sample 1>", "output mentioning Presley"),
                        ("<input sample 2>", "output not mentioning Presley"),
                    ]
                },
                "random": {
                    "test_rate": 0.55,
                    "samples": [
                        ("<input random 1>", "output text..."),
                        ("<input random 2>", "output text..."),
                    ]
                },
                "openai-mask": {
                    "test_rate": 0.65,
                    "samples": [
                        ("<input masked 1>", "output text..."),
                    ]
                },
                "openai-token": {
                    "test_rate": 0.48,
                    "samples": [
                        ("<input token-level 1>", "output text..."),
                    ]
                },
            }
        },
        4: {
            "best_subseq": [101, 202, 303, 404],
            "evaluation": {
                "bert": {
                    "test_rate": 0.68,
                    "samples": [
                        ("<input L4/bert 1>", "output text..."),
                    ]
                },
                "random": {
                    "test_rate": 0.60,
                    "samples": [
                        ("<input L4/random 1>", "output text..."),
                    ]
                },
                "openai-mask": {
                    "test_rate": 0.72,
                    "samples": [
                        ("<input L4/mask 1>", "output text..."),
                    ]
                },
                "openai-token": {
                    "test_rate": 0.55,
                    "samples": [
                        ("<input L4/token 1>", "output text..."),
                    ]
                },
            }
        },
        5: {
            "best_subseq": [101, 202, 303, 404, 505],
            "evaluation": {
                "bert": {
                    "test_rate": 0.80,
                    "samples": [
                        ("<input L5/bert 1>", "output text..."),
                        ("<input L5/bert 2>", "output text..."),
                    ]
                },
                "random": {
                    "test_rate": 0.62,
                    "samples": [
                        ("<input L5/random 1>", "output text..."),
                    ]
                },
                "openai-mask": {
                    "test_rate": 0.58,
                    "samples": []
                },
                "openai-token": {
                    "test_rate": 0.50,
                    "samples": [
                        ("<input L5/token 1>", "output text..."),
                    ]
                },
            }
        }
    }

    # Create an instance of HTMLReporter
    reporter = HTMLReporter(
        model_name=model_name,
        num_perturbations=num_perturbations,
        perturbation_rate=perturbation_rate,
        subsequence_length=subsequence_length,
        prompt_text=prompt_text,
        output_text="",
        target_string=target_string,
        perturbed_samples=perturbed_samples,
        subseqs_at_levels=subseqs_at_levels,
        eval_summaries=eval_summaries
    )

    # Generate the HTML content
    html_report = reporter.generate_html_report()

    # Optionally, write to an HTML file
    with open("../vis/subsequence_analysis_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)

    print("HTML report generated and saved as subsequence_analysis_report.html")

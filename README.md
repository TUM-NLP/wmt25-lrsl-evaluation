# WMT25 Shared Task - LLMs with Limited Resources for Slavic Languages: MT and QA (WMT25-LRSL)

This is a fork of the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to aid with development for the WMT25-LRSL shared task.
This is repository shows how the baseline scores were obtained, and may optionally be used by participants to evaluate their models during the development phase.

## Setup

1. Clone the repository:
```bash
git clone --depth 1 https://github.com/Leukas/wmt25-lrsl-evaluation
```
2. Remember to set up a fresh Python environment with your favourite package manager.
   The baselines were run with Python 3.12, but version 3.10 also works.
3. Install the package:
```bash
cd wmt25-lrsl-evaluation
pip install -e .
```

4. (Temporary) Then clone the data repository into the root folder (`lm-evaluation-harness/`). (This is only temporary, I will clone the data into this repo once it is public.)
5. (Temporary) Run `python prepare_data.py` (Also temporary, I will just provide the data already prepared.)

## Running the baselines

With the package installed, you can run a model on all of the evaluation sets:
```
lm_eval --model hf \
    --model_args pretrained=unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit \
    --tasks sorbian \
    --device cuda:0 \
    --batch_size 8 \
    --output_path baseline_output_sorbian \
    --log_samples
```

This will run all of the Sorbian tasks, and give scores like this:
```
|   Tasks    |Version|  Filter   |n-shot|Metric|   | Value |   |Stderr|
|------------|------:|-----------|-----:|------|---|------:|---|-----:|
|sorbian     |      0|none       |      |acc   |↑  | 0.4832|±  |0.0269|
| - deu-dsb  |      0|remove_tags|     0|bleu  |↑  | 0.5768|±  |0.1207|
|            |       |remove_tags|     0|chrf++|↑  |11.9169|±  |0.1488|
| - deu-hsb  |      0|remove_tags|     0|bleu  |↑  | 0.7668|±  |0.1519|
|            |       |remove_tags|     0|chrf++|↑  |13.3062|±  |0.1416|
| - dsbqa    |      0|none       |      |acc   |↑  | 0.4671|±  |0.0385|
|  - dsbqa-a1|      0|none       |     0|acc   |↑  | 0.4333|±  |0.0920|
|  - dsbqa-a2|      0|none       |     0|acc   |↑  | 0.7143|±  |0.0869|
|  - dsbqa-b1|      0|none       |     0|acc   |↑  | 0.3636|±  |0.0734|
|  - dsbqa-b2|      0|none       |     0|acc   |↑  | 0.3571|±  |0.0646|
| - hsbqa    |      0|none       |      |acc   |↑  | 0.4993|±  |0.0375|
|  - hsbqa-a1|      0|none       |     0|acc   |↑  | 0.7000|±  |0.0851|
|  - hsbqa-a2|      0|none       |     0|acc   |↑  | 0.6429|±  |0.0922|
|  - hsbqa-b1|      0|none       |     0|acc   |↑  | 0.3864|±  |0.0743|
|  - hsbqa-b2|      0|none       |     0|acc   |↑  | 0.2679|±  |0.0597|
```

You can also run each sub-task individually, or in groups by passing them in comma-separated to the `tasks` flag. 
Similarly for Ukrainian, pass in `ukrainian` for all the Ukrainian tasks:
```
|  Tasks   |Version|  Filter   |n-shot|Metric|   | Value |   |Stderr|
|----------|------:|-----------|-----:|------|---|------:|---|-----:|
|ukrainian |      0|none       |      |acc   |↑  | 0.3018|±  |0.0186|
| - cze-ukr|      0|remove_tags|     0|bleu  |↑  | 6.8134|±  |0.1512|
|          |       |remove_tags|     0|chrf++|↑  |27.2625|±  |0.2568|
| - eng-ukr|      0|remove_tags|     0|bleu  |↑  | 8.2124|±  |0.1674|
|          |       |remove_tags|     0|chrf++|↑  |27.0139|±  |0.2574|
| - ukrqa  |      0|none       |     0|acc   |↑  | 0.3018|±  |0.0186|
```

## Inspecting the outputs

If you provided the `--output_path` and `--log_samples` flag, you will find the full examples, with the input and output, in `<output_path>/<model_name>/samples_<task_name>_<timestamp>.jsonl`


## Customization

Each task has a corresponding `.yaml` file that defines the task, found in `lm_eval/tasks/wmt25-lrsl/`. 

For example, here is the `deu-hsb.yaml`:
```yaml
task: deu-hsb
dataset_path: csv
dataset_name: null
dataset_kwargs:
  data_files:  
    test: llms-limited-resources2025-private/Sorbian/hsb/MT/dev.de-hsb.csv
test_split: test
doc_to_text: "Translate the following German text to Upper Sorbian. Put it in this format <hsb> Upper Sorbian translation </hsb>.\n<deu> {{de}} </deu>"
doc_to_target: "{{hsb}}"
filter_list:
  - name: "remove_tags"
    filter:
      - function: "regex"
        regex_pattern: "<hsb> (.*) </hsb>"
      - function: "take_first"
metric_list:
  - metric: bleu
  - metric: chrf++
metadata:
  version: 0.0
```

### Changing the prompt

The `doc_to_text` field in the `.yaml` file prompts the model to put pseudo-html tags around the translation, and then the translation is post-processed with `filter_list` to only consider anything inside these tags. (We do this because LLMs like to output other text, such as explanations.) The output is compared to `doc_to_target` for metrics. `{{de}}` and `{{hsb}}` refer to columns in the CSV. 

### Reducing the development set size

You might notice that the MT evaluation takes a while, so for development purposes you can evaluate on a smaller dev set by adding the flag `--limit {n}` to the command line arguments.

## Further Information

More information can be found in the original [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repo. The final evaluation will be done on outputs you submit, so you have full control over any pre-processing, prompting, and post-processing. 

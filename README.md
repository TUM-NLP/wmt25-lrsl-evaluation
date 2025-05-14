# WMT25 Shared Task - LLMs with Limited Resources for Slavic Languages: MT and QA (WMT25-LRSL)

This is a fork of the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) specifically to aid with evaluation for the WMT25-LRSL shared task. You are in no way required to use this for the shared task. This is merely to help in evaluating models, and so you can see how the baseline scores were obtained.

## Usage

1. First install the package:
```bash
git clone --depth 1 https://github.com/Leukas/wmt25-lrsl-evaluation
cd wmt25-lrsl-evaluation
pip install -e .
``` 
2. (Temporary) Then clone the data repository into the root folder (`lm-evaluation-harness/`). (This is only temporary, I will clone the data into this repo once it is public.)
3. (Temporary) Run `python prepare_data.py` (Also temporary, I will just provide the data already prepared.)
4. Now you can run a model on all of the evaluation sets:
```
lm_eval --model hf 
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

### Inspecting outputs
As long as you provided the `--output_path` and `--log_samples` flag, you will find the full example, with the input and output, in `<output_path>/<model_name>/samples_<task_name>_<timestamp>.jsonl`


## Customization
In the default setup, there is a predefined prompt for the models. The prompts can be modified in the `.yaml` files that denote the task, found in `lm_eval/tasks/wmt25-lrsl/`. 
For example, here is an except from `deu-hsb.yaml`:
```yaml
doc_to_text: "Translate the following German text to Upper Sorbian. Put it in this format <hsb> Upper Sorbian translation </hsb>.\n<deu> {{de}} </deu>"
doc_to_target: "{{hsb}}"
filter_list:
  - name: "remove_tags"
    filter:
      - function: "regex"
        regex_pattern: "<hsb> (.*) </hsb>"
      - function: "take_first"
```

This prompts the model to put pseudo-html tags around the translation, and then the translation is post-processed to only consider anything inside these tags. (We do this because LLMs like to output other text, such as explanations.) 

You are welcome to change these `.yaml` files however you see fit. More information can be found in the original [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repo. The final evaluation will be done on outputs you submit, so you have full control over any pre-processing, prompting, and post-processing. 

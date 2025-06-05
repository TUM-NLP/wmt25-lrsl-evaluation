import argparse
import json
import os

import numpy as np

LANGCODES_3_TO_2 = {
    "eng": "en",
    "deu": "de",
    "ukr": "uk",
    "cze": "cs",
    "hsb": "hsb",
    "dsb": "dsb"
}


def main(eval_harness_outputs, model_name, predictions_save_folder):
    read_folder = os.path.join(eval_harness_outputs, model_name)
    os.makedirs(predictions_save_folder, exist_ok=True)

    for filename in os.listdir(read_folder):
        if filename.endswith(".jsonl"):
            eval_harness_outputs_path = os.path.join(read_folder, filename)
            task_name = filename.split("_")[1]
            predictions_file_path = os.path.join(predictions_save_folder, task_name + "_preds.jsonl")

            is_qa = False
            is_mt = False

            if "qa" in task_name:
                is_qa = True
            else:
                is_mt = True
                src_lang = LANGCODES_3_TO_2[task_name.split("-")[0]]
                tgt_lang = LANGCODES_3_TO_2[task_name.split("-")[1]]
                predictions_file_path = os.path.join(predictions_save_folder, f"{src_lang}-{tgt_lang}_preds.jsonl")

            with (open(eval_harness_outputs_path, 'r', encoding="utf-8") as eval_harness_file,
                  open(predictions_file_path, 'w', encoding="utf-8") as predictions_save_file):
                for line in eval_harness_file:
                    line_data = json.loads(line)

                    if is_mt:
                        src_lang = LANGCODES_3_TO_2[task_name.split("-")[0]]
                        pseudo_test_data = {
                            "doc_id": line_data["doc_id"],
                            "pred": line_data["filtered_resps"][0],
                            "source": line_data["doc"][src_lang]  # not strictly needed but helpful for manual checking
                        }

                    elif is_qa:

                        arguments_run = line_data["arguments"]
                        tried_answers = []
                        for i in range(len(arguments_run)):
                            tried_answers.append(arguments_run[f'gen_args_{i}']['arg_1'])

                        responses = line_data["filtered_resps"]
                        responses = [resp[0] for resp in responses]
                        if "sb" in task_name:
                            num_answers = len(line_data["doc"]["possible_answers"].split("\n"))
                            responses = responses[:num_answers]
                            tried_answers = [int(resp) for resp in tried_answers]
                        elif "ukr" in task_name:
                            tried_answers = [resp.strip() for resp in tried_answers]

                        pred_idx = np.argmax(responses)

                        if pred_idx is not None:
                            prediction = tried_answers[pred_idx]
                        else:
                            prediction = None

                        pseudo_test_data = {
                            "doc_id": line_data["doc_id"],
                            "question_id": line_data["doc"]["question_id"] if "question_id" in line_data["doc"] else None,
                            "pred": prediction,
                            # these two are not strictly needed but helpful for manual inspection
                            "question": line_data["doc"]["question"],
                            "possible_answers": line_data["doc"]["possible_answers"].split("\n")
                        }
                    predictions_save_file.write(json.dumps(pseudo_test_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert lm-evalharness outputs to expected submission format.")
    parser.add_argument("--eval_harness_outputs", type=str, default="../baseline_output_sorbian")
    parser.add_argument("--model_name", type=str, default="unsloth__Qwen2.5-3B-Instruct-unsloth-bnb-4bit")
    parser.add_argument("--predictions_save_folder", type=str, default="submission_predictions")

    args = parser.parse_args()
    main(args.eval_harness_outputs, args.model_name, args.predictions_save_folder)


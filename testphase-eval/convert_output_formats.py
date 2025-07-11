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

    hsb_track, dsb_track = False, False
    for filename in os.listdir(read_folder):
        if filename.endswith(".jsonl"):
            print(f'Reading: {filename}')
            eval_harness_outputs_path = os.path.join(read_folder, filename)
            task_name = filename.split("_")[1]
            dataset_type = task_name.split('-')[-1]
            predictions_file_path = os.path.join(predictions_save_folder, task_name + "_preds.jsonl")

            is_qa = False
            is_mt = False

            if 'qa' in task_name:
                is_qa = True
                lang = filename.split('qa')[0][-3:]
                # print(f'QA for {lang}')
                if hsb_track or lang == "hsb":
                    hsb_track = True
                elif dsb_track or lang == "dsb":
                    dsb_track = True
            else:
                is_mt = True
                src_lang = LANGCODES_3_TO_2[task_name.split("-")[0]]
                tgt_lang = LANGCODES_3_TO_2[task_name.split("-")[1]]
                lang_pair = f'{src_lang}-{tgt_lang}'
                predictions_file_path = os.path.join(predictions_save_folder, f"{lang_pair}_preds.jsonl")

            with (open(eval_harness_outputs_path, 'r', encoding="utf-8") as eval_harness_file,
                  open(predictions_file_path, 'w', encoding="utf-8") as predictions_save_file):
                
                i = 1
                for line in eval_harness_file:
                    line_data = json.loads(line)

                    if is_mt:
                        src_lang = LANGCODES_3_TO_2[task_name.split("-")[0]]
                        if tgt_lang == 'uk' and dataset_type == 'test': # Ukrainian language pair detected: test set
                            source_data = line_data["doc"]["src_text"]
                        elif tgt_lang == 'uk': # Ukrainian language pair detected: dev set
                            source_data = line_data["doc"][src_lang]
                        elif 'sb' in tgt_lang: # Sorbian language pair detected
                            source_data = line_data["doc"][src_lang]
                        else:
                            raise ValueError(f'Unexpected target language: {tgt_lang}')
                        
                        output_pred_data = {
                            "dataset_id": f'wmtslavicllm2025_{lang_pair}',
                            "sent_id": f'{lang_pair}-{i:05}', #line_data["doc_id"],
                            "source": source_data, #line_data["doc"][src_lang], # not strictly needed but helpful for manual checking
                            "pred": line_data["filtered_resps"][0]
                        }
                        i += 1

                    elif is_qa:

                        arguments_run = line_data["arguments"]
                        tried_answers = []
                        for j in range(len(arguments_run)):
                            tried_answers.append(arguments_run[f'gen_args_{j}']['arg_1'])

                        responses = line_data["filtered_resps"]
                        responses_val = [float(resp[0]) for resp in responses]

                        if "sb" in task_name:
                            tried_answers = [int(resp) for resp in tried_answers]
                        elif "ukr" in task_name:
                            tried_answers = [resp.strip() for resp in tried_answers]

                        pred_idx = np.argmax(responses_val)

                        if pred_idx is not None:
                            prediction = tried_answers[pred_idx]
                        else:
                            prediction = None

                        output_pred_data = {
                            "dataset_id": f'wmtslavicllm2025_qa_{lang}',
                            # "doc_id": line_data["doc_id"],
                            "question_id": line_data["doc"]["question_id"] if "question_id" in line_data["doc"] else f'question-{i:04}', #None,
                            # these two are not strictly needed but helpful for manual inspection
                            "question": line_data["doc"]["question"],
                            # "possible_answers": line_data["doc"]["possible_answers"].split("\n")
                            "pred": prediction,
                        }
                        i += 1

                    predictions_save_file.write(json.dumps(output_pred_data, ensure_ascii=False) + "\n")

    # Merging Sorbian QA files for Ocelot submission
    print(hsb_track, dsb_track) 
    sb_qa_levels = ['a1', 'a2', 'b1', 'b2'] #, 'c1']
    file_suffix = ''
    if (hsb_track or dsb_track) and dataset_type == 'test':
        file_suffix = '-test'
        sb_qa_levels += ['c1']
        print(sb_qa_levels)
    if hsb_track:
        with open(os.path.join(predictions_save_folder, f'hsb_qa_preds.jsonl'), 'w') as full_predictions_save_file:
            for level in sb_qa_levels:
                with open(os.path.join(predictions_save_folder, f'hsbqa-{level}{file_suffix}_preds.jsonl')) as inter_file:
                    full_predictions_save_file.write(inter_file.read())
    if dsb_track:
        with open(os.path.join(predictions_save_folder, f'dsb_qa_preds.jsonl'), 'w') as full_predictions_save_file:
            for level in sb_qa_levels:
                with open(os.path.join(predictions_save_folder, f'dsbqa-{level}{file_suffix}_preds.jsonl')) as inter_file:
                    full_predictions_save_file.write(inter_file.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert lm-evalharness outputs to expected submission format.")
    parser.add_argument("--eval_harness_outputs", type=str, default="../baseline_output_sorbian")
    parser.add_argument("--model_name", type=str, default="unsloth__Qwen2.5-3B-Instruct-unsloth-bnb-4bit")
    parser.add_argument("--predictions_save_folder", type=str, default="submission_predictions")

    args = parser.parse_args()
    main(args.eval_harness_outputs, args.model_name, args.predictions_save_folder)
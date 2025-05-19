# prepare_data.py
import pandas as pd
import json

def make_csv(file1, file2, header1, header2):
    with open(file1, "r", encoding="utf-8") as f:
        a_lines = [line.strip() for line in f]

    with open(file2, "r", encoding="utf-8") as f:
        b_lines = [line.strip() for line in f]

    new_fp = file1.split(".")
    new_fp[-1] = "csv"
    new_fp = ".".join(new_fp)
    pd.DataFrame({header1: a_lines, header2: b_lines}).to_csv(new_fp, index=False)

def prep_ukrqa_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for entry in data:
        new_entry = {}
        if len(entry["correct_answers"]) > 1:
            print(entry)
            assert False
        
        possible_answers = []
        for answer in entry["answers"]:
            possible_answers.append(answer["marker"] + ". " + answer["text"])

        new_entry["question"] = entry["question"]
        new_entry["possible_answers"] = " ".join(possible_answers)
        new_entry["correct_answer"] = entry["correct_answers"][0]
        new_data.append(new_entry)

    new_fp = json_file.split(".")
    new_fp[-1] = "csv"
    new_fp = ".".join(new_fp)
    pd.DataFrame(new_data).to_csv(new_fp, index=False)

# put sorbian mt data into csv
make_csv("./llms-limited-resources2025/Sorbian/dsb/MT/dev.de-dsb.de", "./llms-limited-resources2025/Sorbian/dsb/MT/dev.de-dsb.dsb", "de", "dsb")
make_csv("./llms-limited-resources2025/Sorbian/hsb/MT/dev.de-hsb.de", "./llms-limited-resources2025/Sorbian/hsb/MT/dev.de-hsb.hsb", "de", "hsb")


# put ukrainian mt data into csv
make_csv("./llms-limited-resources2025/Ukrainian/MT/dev.en-uk.en", "./llms-limited-resources2025/Ukrainian/MT/dev.en-uk.uk", "en", "uk")
make_csv("./llms-limited-resources2025/Ukrainian/MT/dev.cs-uk.cs", "./llms-limited-resources2025/Ukrainian/MT/dev.cs-uk.uk", "cs", "uk")

# put ukrainian qa data into csv
prep_ukrqa_data("./llms-limited-resources2025/Ukrainian/QA/dev.json")
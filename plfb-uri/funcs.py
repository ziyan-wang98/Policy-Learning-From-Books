import json
import numpy as np
import zipfile

def load_jsonl_list(file_path):
    res_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            res_list.append(json.loads(line))
    return res_list


def npz_extractor(npz_file_path, key_check_list=['obs_before_modified_by_acs', 'action']):
    file = npz_file_path
    try:
        res = np.load(file, allow_pickle=True)
        for key in key_check_list:
            if key not in res.keys():
                print("skip this file", file, "missing key", key)
                print("key list", list(res.keys()))
                return None
    except zipfile.BadZipFile as e:
        print("BadZipFile, skip this file", file)
        print("error", e)
        return None 
    return res

def json_str_clean(input_str, single_line=False):
    if single_line:
        return input_str.replace("```json\n", "").replace("\n```", "").replace("\n", "").replace("```", "").replace(",}", "}")
    else:
        return input_str.replace("\"\"\"\n", "\"").replace("\n\"\"\"\"", "\"").replace(",\n", ",").replace("{\n", "{").replace("\n}", "}").replace("```json\n", "").replace("\n```", "").replace("```", "").replace("\n", "\\n")
    
def formulate_param_to_name(params_dict):
    return  '&'.join([f'{k}={v}'.replace(" ", "--") for k, v in params_dict.items()])

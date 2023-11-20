# -*- coding: utf-8 -*-

import json

data = []


def read_lines(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [l_.strip() for l_ in lines]


def write_txt_2_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        res_dict = json.load(json_file)
    return res_dict


def write_json_2_file(json_, file_path):
    with open(file_path, 'w') as f:
        f.write(json.dumps(json_, ensure_ascii=False))

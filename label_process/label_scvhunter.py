import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

def supplement_ext():
    file_dir = './datasets/scvhunter/source/blockinfo/timestamp/undependency'
    for filename in os.listdir(file_dir):
        file_path = os.path.join(file_dir, filename)
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            if not ext:
                new_filename = filename + ".sol"
                os.rename(file_path, os.path.join(file_dir, new_filename))

def count_sol_files():
    root_dir = './datasets/scvhunter/source'
    sol_folders = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        count = 0
        for file in filenames:
            if file.endswith(".sol"):
                count += 1
        if count > 0:
            sol_folders[dirpath] = count
    for folder, num in sol_folders.items():
        print(f"{folder}: {num}")

def find_label(file_dir, save_file_dir):
    labels = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if not file.endswith(".sol"):
                continue
            file_path = os.path.join(dirpath, file)
            solc_versions = get_versions_from_file(file_path)
            solc_version = check_compile(file_path, solc_versions)
            if solc_version is None:
                print(f"{file} can't compile")
                continue
            if dirpath.split('/')[-1] == 'dependency':
                label = 1
            else:
                label = 0
            labels.append({
                'file_path': file_path,
                'label': label,
                'solc_version': solc_version
            })
            print(f'{file} compile success')
    scv_type = file_dir.split('/')[-1]
    os.makedirs(save_file_dir, exist_ok=True)
    save_file_path = os.path.join(save_file_dir, f'{scv_type}.json')
    with open(save_file_path, 'w') as f:
        json.dump(labels, f, indent=4)
    

if __name__ == '__main__':
    scv = 'timestamp'
    sys.stdout = Logger(f'./logs/label/scvhunter/{scv}.txt', sys.stdout)
    file_dir = f'../datasets/scvhunter/source/blockinfo/{scv}'
    save_file_dir = f'../datasets/scvhunter/labels'
    find_label(file_dir, save_file_dir)
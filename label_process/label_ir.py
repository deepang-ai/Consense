import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


def clear_labels(file_dir, source_label_path, sheet_name) -> List[dict]:
    '''
    label process:
    1. remove no exist file
    2. remove cannot compile file

    Args:
        file_dir: contract source file directory
        source_label_path: origin label path
        sheet_name: vulunerable type
    '''
    scv_labels = pd.ExcelFile(source_label_path).parse(sheet_name).iloc[:, :3]
    nums = len(scv_labels)
    final_labels = []
    can_compile_list = []
    cannot_list = []
    can_file_versions = []
    for i in range(nums):
        file_name = scv_labels['file'][i]
        file_path = os.path.join(file_dir, f'{file_name}.sol')
        # file not fund
        if not os.path.exists(file_path):
            print(f'[{i+1}/{nums}] {sheet_name} {file_name} not exist!')
            continue
        # can compile
        if file_name in can_compile_list:
            final_labels.append({
                'file': scv_labels['file'][i],
                'contract': scv_labels['contract'][i],
                'label': convert_to_int(scv_labels['ground truth'][i]),
                'solc_version': can_file_versions[can_compile_list.index(file_name)]
            })
            continue
        # cannot compile
        elif file_name in cannot_list:
            continue
        # get compile version
        solc_versions = get_versions_from_file(file_path)
        solc_version = check_compile(file_path, solc_versions)
        # cannot compile
        if solc_version is None:
            cannot_list.append(file_name)
            print(f"[{i+1}/{nums}] can't compile: {sheet_name} {file_name} {solc_versions}")
            continue
        # can compile
        can_compile_list.append(file_name)
        can_file_versions.append(solc_version)
        print(f'[{i+1}/{nums}] {sheet_name} {file_name} compile success')
        final_labels.append({
            'file': scv_labels['file'][i],
            'contract': scv_labels['contract'][i],
            'label': scv_labels['ground truth'][i],
            'solc_version': solc_version
        })
    return final_labels


def main(origin_label_path, save_file_dir):
    scv_types = ['BN', 'DE', 'EF', 'SE', 'OF', 'RE', 'TP', 'UC']
    scv_full_names = ['block number dependency', 'dangerous delegatecall',
                        'ether frozen', 'ether strict equality',
                        'integer overflow', 'reentrancy',
                        'timestamp dependency','unchecked external call']
    contract_dfs = []
    for index, scv in  enumerate(scv_types):
        file_dir = os.path.join(os.path.dirname(origin_label_path), f'{scv_full_names[index]} ({scv})')
        contract_labels = clear_labels(file_dir, origin_label_path, scv_full_names[index])
        contract_df = pd.DataFrame(contract_labels)
        contract_dfs.append(contract_df)
    os.makedirs(save_file_dir, exist_ok=True)
    contract_label_path = os.path.join(save_file_dir, 'contract_labels.xlsx')
    # save contract labels
    with pd.ExcelWriter(contract_label_path) as writer:
        for index, contract_df in enumerate(contract_dfs):
            contract_df.to_excel(writer, sheet_name=scv_types[index], index=False)

def label_count(label_path):
    labels = pd.ExcelFile(label_path)
    sheet_names = labels.sheet_names
    for sheet_name in sheet_names:
        v_nums = 0
        nv_nums = 0
        df = pd.read_excel(labels, sheet_name=sheet_name)
        for value in df['label']:
            if convert_to_int(value) == 1:
                v_nums += 1
            else:
                nv_nums += 1
        print(f"{sheet_name}\t\tnums: {len(df)}, vulunerable: {v_nums}, safe:{nv_nums}")


if  __name__ == '__main__': 
    # sys.stdout = Logger('./logs/ir/log.txt', sys.stdout)
    # original_label_path = '../datasets/IR-ESCD/source/ground truth label.xlsx'
    # save_label_dir = '../datasets/IR-ESCD/label'
    # main(original_label_path, save_label_dir)

    label_path = '../datasets/IR-ESCD/label/contract_labels.xlsx'
    label_count(label_path)
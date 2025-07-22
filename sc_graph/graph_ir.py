import os
import sys
import csv
import time
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import LlamaForCausalLM, CodeLlamaTokenizer


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sc_graph.parse import *

device = 'cuda'


def llama_embedding(sentences:str, model, tokenizer):
    '''
    return the function or variable embedding using codeLlama
    '''
    model.eval()
    embeddings = []
    for sentence in sentences:
        with torch.no_grad():
            encoding = tokenizer(sentence, padding=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            model_output = model(input_ids, attention_mask, output_hidden_states=True)
            data = model_output.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
            masked_embeddings = data * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)    # 这里表达样本对所有token位置的隐藏理解之和，如果除以hidden_size倍可以代表平均
            seq_length = torch.sum(mask, dim=1) # 这部分表示每个样本的序列长度的hidden_size倍
            embedding = sum_embeddings / seq_length
            normalized_embeddings = F.normalize(embedding, p=2, dim=1)  # p=2： 表示使用L2范数归一化（标准化或单位化）
            ret = normalized_embeddings.squeeze(0).cpu().tolist()
            embeddings.append(ret)
    return embeddings


def write_empty_csv(save_dir):
    with open(os.path.join(save_dir, 'num-node-list.csv'), mode='w', newline='', encoding='utf-8') as file:
        pass
    with open(os.path.join(save_dir, 'num-edge-list.csv'), mode='w', newline='', encoding='utf-8') as file:
        pass
    with open(os.path.join(save_dir, 'graph-label.csv'), mode='w', newline='', encoding='utf-8') as file:
        pass
    with open(os.path.join(save_dir, 'edge.csv'), mode='w', newline='', encoding='utf-8') as file:
        pass
    with open(os.path.join(save_dir, 'node-feat.csv'), mode='w', newline='', encoding='utf-8') as file:
        pass


def load_model(model_path):
    # load model
    model = LlamaForCausalLM.from_pretrained(model_path,device_map = 'auto').half()
    tokenizer = CodeLlamaTokenizer.from_pretrained(model_path)

    # pad token
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main(source_dir:str, label_path:str, model_path:str, save_dir:str):
    scv_types = ['BN', 'DE', 'EF', 'SE', 'OF', 'RE', 'TP', 'UC']
    scv_full_names = ['block number dependency', 'dangerous delegatecall',
                        'ether frozen', 'ether strict equality',
                        'integer overflow', 'reentrancy',
                        'timestamp dependency','unchecked external call']
    for index, scv in  enumerate(scv_types):
        file_dir = os.path.join(source_dir, f'{scv_full_names[index]} ({scv})')
        save_dir = os.path.join(save_dir, scv, 'raw')
        os.makedirs(save_dir, exist_ok=True)
        write_empty_csv(save_dir)
        start_time = time.time()
        model, tokenizer = load_model(model_path)
        df = pd.read_excel(label_path, sheet_name=scv)
        for index, row in df.iterrows():
            file_name = row['file']
            contract_name = row['contract']
            label = row['label']
            solc_version = row['solc_version']
            file_path = os.path.join(file_dir, f'{file_name}.sol')
            try:
                solidity_parser = SolidityParser(file_path, solc_version)
            except:
                print('Parse error:', file_path)
                continue
            res = contract_graph(solidity_parser)
            if res is None or len(re[0])==0:
                print(f'contract is empty: {file_path}_{contract_name}')
                continue
            nodes, edges = res
            sentences = []
            for node in res[0]:
                sentences.append(solidity_parser.get_node(node).content)
            if len(sentences) != len(nodes):
                print(f'Get tokens failed: {file_path}_{contract_name}')
                continue
            embeddings = llama_embedding(sentences, model, tokenizer)
            with open(os.path.join(save_dir, 'num-node-list.csv'), mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([len(nodes)])
            with open(os.path.join(save_dir, 'num-edge-list.csv'), mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([len(edges)])
            with open(os.path.join(save_dir, 'graph-label.csv'), mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([label])
            with open(os.path.join(save_dir, 'edge.csv'), mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for edge in edges:
                    writer.writerow([int(edge[0]), int(edge[1])])
            with open(os.path.join(save_dir, 'node-feat.csv'), mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for embedding in embeddings:
                    writer.writerow(embedding)
                embeddings = []
            print(f'[{index+1}/{len(df)}] {os.path.basename(file_path)} done!')
        avg_time = (time.time() - start_time)/len(df)
        print(f"count: {len(df)}, avg time: {avg_time:.5f}s") 



if __name__ == '__main__':
    sys.stdout = Logger(f'./logs/ir/log.txt', sys.stdout)
    source_dir = '../datasets/IR-ESCD/source'
    label_path = f'../datasets/IR-ESCD/label/contract_labels.json'
    # code Llama location
    model_path = '../../codellama_contract/'
    save_dir = f'../dataset/IR-ESCD/embeddings'
    main(source_dir, label_path, model_path, save_dir)
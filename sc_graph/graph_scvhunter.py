import os
import sys
import csv
import json
import time
import torch
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
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'data len: {len(data)}')
    save_dir = os.path.join(save_dir, 'raw')
    data_len = len(data)
    os.makedirs(save_dir, exist_ok=True)
    write_empty_csv(save_dir)
    start_time = time.time()
    model, tokenizer = load_model(model_path)
    for i in range(data_len):
        file_path = data[i]['file_path']
        label = data[i]['label']
        solc_version = data[i]['solc_version']
        file_path = os.path.join(source_dir, file_path)
        try:
            solidity_parser = SolidityParser(file_path, solc_version)
        except:
            print('Parse error:', file_path)
            continue
        nodes, edges, sentences = solidity_graph(solidity_parser)
        if len(sentences) != len(nodes):
            print(f'Get tokens failed: {file_path}')
            continue
        embeddings = llama_embedding(sentences, model, tokenizer)
        if len(embeddings) != len(nodes):
            print(f'Embedding failed: {file_path}')
            continue
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
        print(f'[{i+1}/{data_len}] {os.path.basename(file_path)} done!')
    avg_time = (time.time() - start_time)/data_len
    print(f"count: {data_len}, avg time: {avg_time:.5f}s") 



if __name__ == '__main__':
    # scv_name = 'reentrancy'
    # scv_name = 'timestamp'
    scv_name = 'origin'
    sys.stdout = Logger(f'./logs/scvhunter/{scv_name}.txt', sys.stdout)
    source_dir = '../dataset/scvhunter/source'
    label_path = f'../dataset/scvhunter/labels/{scv_name}.json'
    # code Llama location
    model_path = '../../codellama_contract/'
    save_dir = f'../dataset/scvhunter/embeddings/{scv_name}'
    main(source_dir, label_path, model_path, save_dir)
import pandas as pd
import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops


class PygGraphDataset(InMemoryDataset):
    def __init__(self, root, name, add_inverse_edge=False, add_self_loop=False):
        """
        root = Where the dataset should be stored. This folder is split into:
            - raw/
            - processed/
        """
        self.root = os.path.join(root, name)
        self.add_inverse_edge = add_inverse_edge
        self.add_self_loop = add_self_loop
        self._num_classes = 2
        super(PygGraphDataset, self).__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['pyg_graph_data.pt']
    
    @property
    def num_classes(self):
        return self._num_classes

    def download(self):
        pass

    def process(self):
        raw_dir = os.path.join(self.root, 'raw')
        num_node_list = pd.read_csv(os.path.join(raw_dir, 'num-node-list.csv'), header=None).iloc[:, 0].tolist()
        num_edge_list = pd.read_csv(os.path.join(raw_dir, 'num-edge-list.csv'), header=None).iloc[:, 0].tolist()
        graph_label_list = pd.read_csv(os.path.join(raw_dir, 'graph-label.csv'), header=None).iloc[:, 0].tolist()

        edges = pd.read_csv(os.path.join(raw_dir, 'edge.csv'), header=None).values.T.astype(np.int64)
        node_feats = pd.read_csv(os.path.join(raw_dir, 'node-feat.csv'), header=None).values.astype(np.float32)


        graph_list = []
        num_node_accum = 0
        num_edge_accum = 0

        print('Processing graphs...')
        for num_node, num_edge, graph_label in tqdm(zip(num_node_list, num_edge_list, graph_label_list), total=len(num_node_list)):
            
            node_feat = node_feats[num_node_accum:num_node_accum + num_node]
            num_node_accum += num_node

            edge_index = edges[:, num_edge_accum:num_edge_accum + num_edge]
            num_edge_accum += num_edge

            graph = Data(x = torch.from_numpy(node_feat), edge_index = torch.from_numpy(edge_index))

            graph.num_nodes = num_node
            graph.y = torch.tensor([graph_label])

            if self.add_inverse_edge:
                graph.edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)

            if self.add_self_loop:
                graph.edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)

            graph_list.append(graph)

    
        data, slices = self.collate(graph_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])






if __name__ == '__main__':
    root_dir = '../datasets/scvhunter/embeddings'
    name = 'reentrancy'
    dataset = PygGraphDataset(root=root_dir, name=name, add_self_loop=True)
    print(len(dataset))

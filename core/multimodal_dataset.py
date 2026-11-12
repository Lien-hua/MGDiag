import dgl
import torch
from torch.utils.data import Dataset


NODE_TYPE_TO_ID = {"service": 0, "pod": 1, "node": 2, "unknown": 3}


class MultiModalDataSet(Dataset):
    def __init__(self):
        self.data = []

    def add_data(
        self,
        metric_Xs,
        trace_Xs,
        log_Xs,
        global_root_id,
        failure_type_id,
        local_roots,
        nodes,
        node_types,
        edges,
        metadata,
    ):
        node_num = len(nodes)
        if not node_num:
            raise ValueError(f"sample {metadata.get('index')} has an empty graph")
        if edges:
            sources, targets = zip(*edges)
            graph = dgl.graph((list(sources), list(targets)), num_nodes=node_num)
        else:
            graph = dgl.graph(([], []), num_nodes=node_num)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph.ndata["metric"] = torch.as_tensor(metric_Xs, dtype=torch.float32)
        graph.ndata["trace"] = torch.as_tensor(trace_Xs, dtype=torch.float32)
        graph.ndata["log"] = torch.as_tensor(log_Xs, dtype=torch.float32)
        root_labels = torch.zeros(node_num, dtype=torch.long)
        for root in local_roots:
            root_labels[nodes.index(root)] = 1
        graph.ndata["root"] = root_labels
        graph.ndata["node_type"] = torch.tensor(
            [NODE_TYPE_TO_ID.get(node_type, NODE_TYPE_TO_ID["unknown"]) for node_type in node_types],
            dtype=torch.long,
        )
        self.data.append(
            (graph, (global_root_id, failure_type_id), metadata)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

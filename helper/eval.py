import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def RCA_eval(root_logit, num_nodes_list, roots):
    """Evaluate one or more valid root causes per graph.

    HR@K is an any-hit metric. RootRecall@K measures how much of a multi-root
    label is covered by the first K predictions.
    """
    result = {
        **{f"HR@{k}": [] for k in range(1, 6)},
        **{f"RootRecall@{k}": [] for k in range(1, 6)},
        "MRR@3": [],
    }
    start = 0
    for num_nodes in num_nodes_list:
        count = int(num_nodes)
        end = start + count
        logits = root_logit[start:end].flatten()
        positive = set(torch.nonzero(roots[start:end] > 0, as_tuple=False).flatten().tolist())
        if not positive:
            raise ValueError("graph has no root-cause label")
        ranking = torch.argsort(logits, descending=True).tolist()
        for k in range(1, 6):
            predicted = set(ranking[: min(k, count)])
            hits = len(predicted & positive)
            result[f"HR@{k}"].append(float(hits > 0))
            result[f"RootRecall@{k}"].append(hits / len(positive))
        first_rank = min(ranking.index(root) + 1 for root in positive)
        result["MRR@3"].append(1.0 / first_rank if first_rank <= 3 else 0.0)
        start = end
    return {key: float(np.mean(values)) for key, values in result.items()}


def FTI_eval(output, target, k=5):
    predictions = output.argmax(dim=1).cpu().detach().numpy()
    truth = target.cpu().detach().numpy()
    precision = precision_score(truth, predictions, average="weighted", zero_division=0)
    recall = recall_score(truth, predictions, average="weighted", zero_division=0)
    f1 = f1_score(truth, predictions, average="weighted", zero_division=0)
    return {"pre": precision, "rec": recall, "f1": f1}


def target_rank(output, target, k=10):
    k = min(k, output.shape[1])
    predictions = output.topk(k, 1, True, True).indices
    ranks = []
    for row, expected in zip(predictions, target):
        locations = torch.nonzero(row == expected, as_tuple=False).flatten()
        ranks.append(int(locations[0]) + 1 if len(locations) else k + 1)
    return ranks


def precision(output, target, k=5):
    return FTI_eval(output, target, k)["pre"]


def recall(output, target, k=5):
    return FTI_eval(output, target, k)["rec"]


def f1score(output, target, k=5):
    return FTI_eval(output, target, k)["f1"]

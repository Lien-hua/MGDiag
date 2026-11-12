import json
import os
import time
from typing import List

import pandas as pd
from tqdm.auto import tqdm

from config.exp_config import Config
from core.aug import aug_drop_node
from core.multimodal_dataset import MultiModalDataSet
from helper import io_util
from process.events.fasttext_w2v import FastTextEncoder


class EventProcess:
    REQUIRED_RAW_FILES = (
        "metrics.json", "traces.json", "logs.json", "edges.json",
        "nodes.json", "node_types.json",
    )

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.dataset = config.dataset

    def process(self, reconstruct=False):
        started_at = time.time()
        self.data_path = os.path.join("data", self.dataset)
        self._load_processed_data()
        self._validate_processed_data()
        failure_types = sorted(self.labels["anomaly_type"].astype(str).unique())
        self.types = ["normal"] + failure_types
        self.config.failure_types = failure_types
        self.config.ft_num = len(failure_types)
        self.logger.info(
            "Loaded %d samples, %d failure types", len(self.labels), self.config.ft_num
        )
        if reconstruct:
            self.logger.info("[Dataset 2/3] Reconstruct event embeddings")
            self.build_embedding()
        else:
            self.logger.info("[Dataset 2/3] Reuse cached event embeddings")
        self.logger.info("[Dataset 3/3] Build graph datasets")
        datasets = self.build_dataset()
        self.logger.info(
            "Dataset processing finished in %.2f seconds", time.time() - started_at
        )
        return datasets

    def _load_processed_data(self):
        label_path = os.path.join(self.data_path, "label.csv")
        raw_dir = os.path.join(self.data_path, "raw")
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"{label_path} does not exist. Run: python -m extractor.pipeline "
                f"--dataset {self.dataset}"
            )
        self.labels = pd.read_csv(label_path, dtype={"index": str})
        loaded = {}
        for filename in tqdm(
            self.REQUIRED_RAW_FILES,
            desc=f"Load {self.dataset}", unit="file", leave=False,
        ):
            path = os.path.join(raw_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path, "r", encoding="utf-8") as handle:
                loaded[filename] = json.load(handle)
        self.metrics = loaded["metrics.json"]
        self.traces = loaded["traces.json"]
        self.logs = loaded["logs.json"]
        self.edges = loaded["edges.json"]
        self.nodes = loaded["nodes.json"]
        self.node_types = loaded["node_types.json"]
        self.labels["index"] = self.labels["index"].astype(str)
        self.labels["roots"] = self.labels.apply(self._parse_roots, axis=1)

    @staticmethod
    def _parse_roots(row) -> List[str]:
        if "instances_json" in row and pd.notna(row["instances_json"]):
            roots = json.loads(row["instances_json"])
        elif "instance" in row and pd.notna(row["instance"]):
            roots = [row["instance"]]
        else:
            raise ValueError(f"sample {row.get('index')} has no root-cause label")
        if not isinstance(roots, list):
            roots = [roots]
        return [str(root) for root in roots]

    def _validate_processed_data(self):
        required_columns = {"index", "data_type", "anomaly_type", "roots"}
        missing_columns = required_columns - set(self.labels.columns)
        if missing_columns:
            raise ValueError(f"label.csv missing columns: {sorted(missing_columns)}")
        label_ids = set(self.labels["index"])
        mappings = {
            "metrics": self.metrics, "traces": self.traces, "logs": self.logs,
            "edges": self.edges, "nodes": self.nodes,
            "node_types": self.node_types,
        }
        for name, mapping in mappings.items():
            missing = label_ids - set(mapping)
            extra = set(mapping) - label_ids
            if missing or extra:
                raise ValueError(
                    f"{name} keys do not match labels; missing={sorted(missing)[:5]}, "
                    f"extra={sorted(extra)[:5]}"
                )
        for row in self.labels.to_dict("records"):
            case_id = str(row["index"])
            absent = [root for root in row["roots"] if root not in self.nodes[case_id]]
            if absent:
                raise ValueError(f"{case_id}: roots absent from nodes.json: {absent}")

    def _events_for_node(self, modality: str, events, node: str) -> List[str]:
        documents = []
        for raw_event in events:
            event = [str(item) for item in raw_event]
            if not event:
                continue
            if modality == "metric":
                if event[0] != node:
                    continue
                if not self.config.metric_direction and len(event) >= 3:
                    event = event[:-1]
            elif modality == "trace":
                if len(event) < 2 or node not in event[:2]:
                    continue
                if not self.config.trace_op and len(event) >= 3:
                    event = [item for index, item in enumerate(event) if index != 2]
                if not self.config.trace_ab_type and len(event) >= 4:
                    event = event[:-1]
            elif modality == "log":
                if event[0] != node:
                    continue
            else:
                raise ValueError(f"unsupported modality: {modality}")
            documents.append("&".join(event))
        return documents

    def build_embedding(self):
        data_map = {"metric": self.metrics, "trace": self.traces, "log": self.logs}
        all_nodes = sorted({item for values in self.nodes.values() for item in values})
        train_rows = self.labels[self.labels["data_type"] == "train"]
        for modality, data in tqdm(
            data_map.items(), desc="Embedding modalities", unit="modality"
        ):
            encoder = FastTextEncoder(
                modality, all_nodes, self.types,
                embedding_dim=self.config.alert_embedding_dim,
                epochs=5, logger=self.logger,
            )
            documents, labels = [], []
            for row in tqdm(
                train_rows.to_dict("records"), total=len(train_rows),
                desc=f"{modality}: corpus", leave=False,
            ):
                case_id = str(row["index"])
                roots = set(row["roots"])
                anomaly_index = self.types.index(str(row["anomaly_type"]))
                for node in self.nodes[case_id]:
                    documents.append(self._events_for_node(modality, data[case_id], node))
                    label_index = anomaly_index if node in roots else 0
                    labels.append(f"__label__{node}::{label_index}")
            encoder.fit(documents, labels)

            embeddings = {}
            for row in tqdm(
                self.labels.to_dict("records"), total=len(self.labels),
                desc=f"{modality}: embeddings", leave=False,
            ):
                case_id = str(row["index"])
                embeddings[case_id] = [
                    encoder.get_sentence_embedding(
                        self._events_for_node(modality, data[case_id], node)
                    )
                    for node in self.nodes[case_id]
                ]
            tmp_path = os.path.join(self.data_path, "tmp")
            os.makedirs(tmp_path, exist_ok=True)
            io_util.save_pkl(os.path.join(tmp_path, f"{modality}.pkl"), embeddings)

    def build_dataset(self):
        embeddings = {}
        for modality in ("metric", "trace", "log"):
            path = os.path.join(self.data_path, "tmp", f"{modality}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} does not exist. Run main.py with --reconstruct first."
                )
            embeddings[modality] = io_util.load_pkl(path)
        failure_to_index = {
            failure_type: index
            for index, failure_type in enumerate(self.config.failure_types)
        }
        root_keys = sorted({tuple(roots) for roots in self.labels["roots"]})
        root_set_to_index = {root_key: index for index, root_key in enumerate(root_keys)}
        train_data, test_data = MultiModalDataSet(), MultiModalDataSet()

        for row in tqdm(
            self.labels.to_dict("records"), total=len(self.labels),
            desc="Build graphs", unit="sample", leave=False,
        ):
            case_id = str(row["index"])
            roots = list(row["roots"])
            nodes = self.nodes[case_id]
            type_mapping = self.node_types[case_id]
            fallback_type = row.get("instance_type", "unknown")
            metadata = {
                "index": case_id,
                "nodes": nodes,
                "node_types": [type_mapping.get(node, "unknown") for node in nodes],
                "roots": roots,
                "root_types": [type_mapping.get(root, fallback_type) for root in roots],
                "failure_type": str(row["anomaly_type"]),
            }
            target = train_data if row["data_type"] == "train" else test_data
            target.add_data(
                metric_Xs=embeddings["metric"][case_id],
                trace_Xs=embeddings["trace"][case_id],
                log_Xs=embeddings["log"][case_id],
                global_root_id=root_set_to_index[tuple(roots)],
                failure_type_id=failure_to_index[str(row["anomaly_type"])],
                local_roots=roots,
                nodes=nodes,
                node_types=metadata["node_types"],
                edges=self.edges[case_id],
                metadata=metadata,
            )

        augmented = []
        if self.config.aug_times > 0:
            for _ in tqdm(range(self.config.aug_times), desc="Augment rounds"):
                for graph, labels, _metadata in train_data:
                    protected_roots = graph.ndata["root"].nonzero().flatten().tolist()
                    augmented.append((
                        aug_drop_node(
                            graph, protected_roots,
                            drop_percent=self.config.aug_percent,
                        ),
                        labels,
                    ))
        self.logger.info(
            "Graph datasets built: train=%d, augmented=%d, test=%d",
            len(train_data), len(augmented), len(test_data),
        )
        return train_data, augmented, test_data

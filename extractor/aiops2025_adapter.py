import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from extractor.base import CaseData, CaseSpec, DatasetAdapter
from extractor.events import derive_trace_calls
from extractor.registry import register_adapter


TIMEZONE = "Asia/Shanghai"
METRIC_COLUMNS = ["time", "entity", "entity_type", "metric", "value"]
TRACE_COLUMNS = [
    "time",
    "duration",
    "trace_id",
    "span_id",
    "parent_span_id",
    "service",
    "pod",
    "node",
    "operation",
    "span_kind",
    "peer_address",
    "status_code",
]
LOG_COLUMNS = ["time", "pod", "node", "message"]


def parse_local_time(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TIMEZONE)
    return timestamp.tz_convert(TIMEZONE)


def parse_time_column(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    return parsed.dt.tz_convert(TIMEZONE)


def in_window(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[(frame["time"] >= start) & (frame["time"] < end)].copy()


@register_adapter("aiops2025")
class AIOps2025Adapter(DatasetAdapter):
    REQUIRED_CASE_FILES = {
        "metadata.json",
        "logs.parquet",
        "trace.parquet",
        "service.parquet",
        "pod.parquet",
        "node.parquet",
    }

    def __init__(self, root: Path):
        super().__init__(root)
        ground_truth_path = self.root / "ground_truth.json"
        if not ground_truth_path.exists():
            raise FileNotFoundError(ground_truth_path)
        with ground_truth_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        self.ground_truth = {row["uuid"]: row for row in rows}
        self.deployments = self._load_deployments(self.root / "deploy.parquet")

    def cases(self) -> List[CaseSpec]:
        result: List[CaseSpec] = []
        for split in ("train", "test"):
            split_dir = self.root / split
            if not split_dir.exists():
                continue
            for case_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                present = {path.name for path in case_dir.iterdir()}
                missing = self.REQUIRED_CASE_FILES - present
                if missing:
                    raise ValueError(
                        f"{case_dir} is missing required files: {sorted(missing)}"
                    )
                if case_dir.name not in self.ground_truth:
                    raise ValueError(f"missing ground truth for {case_dir.name}")
                with (case_dir / "metadata.json").open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                label = self.ground_truth[case_dir.name]
                raw_roots = label["instance"]
                roots = raw_roots if isinstance(raw_roots, list) else [raw_roots]
                result.append(
                    CaseSpec(
                        case_id=case_dir.name,
                        split=split,
                        roots=[str(root) for root in roots],
                        root_type=str(label["instance_type"]).lower(),
                        failure_type=str(label["fault_type"]),
                        normal_start=parse_local_time(metadata["normal"]["start_time"]),
                        normal_end=parse_local_time(metadata["normal"]["end_time"]),
                        abnormal_start=parse_local_time(metadata["abnormal"]["start_time"]),
                        abnormal_end=parse_local_time(metadata["abnormal"]["end_time"]),
                        path=case_dir,
                    )
                )
        if not result:
            raise ValueError(f"no cases found below {self.root}")
        return result

    def load_case(self, spec: CaseSpec) -> CaseData:
        service_metrics = self._wide_metrics(
            spec.path / "service.parquet", "service", "service"
        )
        pod_metrics = self._pod_metrics(spec.path / "pod.parquet")
        node_metrics = self._wide_metrics(
            spec.path / "node.parquet", "node", "node"
        )
        metrics = pd.concat(
            [service_metrics, pod_metrics, node_metrics], ignore_index=True
        )
        traces = self._traces(spec.path / "trace.parquet")
        logs = self._logs(spec.path / "logs.parquet")
        return CaseData(
            spec=spec,
            normal_metrics=in_window(metrics, spec.normal_start, spec.normal_end),
            abnormal_metrics=in_window(metrics, spec.abnormal_start, spec.abnormal_end),
            normal_traces=in_window(traces, spec.normal_start, spec.normal_end),
            abnormal_traces=in_window(traces, spec.abnormal_start, spec.abnormal_end),
            normal_logs=in_window(logs, spec.normal_start, spec.normal_end),
            abnormal_logs=in_window(logs, spec.abnormal_start, spec.abnormal_end),
        )

    def build_topology(
        self, data: CaseData
    ) -> Tuple[List[str], List[List[int]], Dict[str, str]]:
        spec = data.spec
        deployments = self.deployments[
            (self.deployments["start_time"] < spec.abnormal_end)
            & (self.deployments["end_time"] >= spec.normal_start)
        ]
        node_types: Dict[str, str] = {}
        edge_names = set()

        def add_node(name, node_type):
            if pd.isna(name) or not str(name).strip():
                return
            node_types.setdefault(str(name), node_type)

        def add_bidirectional(left, right):
            if pd.isna(left) or pd.isna(right):
                return
            left, right = str(left), str(right)
            if left and right and left != right:
                edge_names.add((left, right))
                edge_names.add((right, left))

        for row in deployments.itertuples(index=False):
            add_node(row.service, "service")
            add_node(row.pod, "pod")
            add_node(row.node, "node")
            add_bidirectional(row.service, row.pod)
            add_bidirectional(row.pod, row.node)

        for frame in (data.normal_metrics, data.abnormal_metrics):
            for row in frame[["entity", "entity_type"]].drop_duplicates().itertuples(index=False):
                add_node(row.entity, row.entity_type)
        for frame in (data.normal_logs, data.abnormal_logs):
            for row in frame[["pod", "node"]].drop_duplicates().itertuples(index=False):
                add_node(row.pod, "pod")
                add_node(row.node, "node")
                add_bidirectional(row.pod, row.node)
        traces = pd.concat([data.normal_traces, data.abnormal_traces], ignore_index=True)
        for row in traces[["service", "pod", "node"]].drop_duplicates().itertuples(index=False):
            add_node(row.service, "service")
            add_node(row.pod, "pod")
            add_node(row.node, "node")
            add_bidirectional(row.service, row.pod)
            add_bidirectional(row.pod, row.node)
        for row in derive_trace_calls(traces)[["src", "dst"]].drop_duplicates().itertuples(index=False):
            add_node(row.src, "service")
            add_node(row.dst, "service")
            add_bidirectional(row.src, row.dst)

        for root in spec.roots:
            add_node(root, spec.root_type)

        order = {"service": 0, "pod": 1, "node": 2}
        nodes = sorted(node_types, key=lambda name: (order.get(node_types[name], 99), name))
        node_to_index = {node: index for index, node in enumerate(nodes)}
        edges = sorted(
            [node_to_index[left], node_to_index[right]]
            for left, right in edge_names
            if left in node_to_index and right in node_to_index
        )
        return nodes, edges, node_types

    @staticmethod
    def _load_deployments(path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        required = {"pod", "service", "node", "start_time", "end_time"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = frame[list(required)].copy()
        frame["start_time"] = parse_time_column(frame["start_time"])
        frame["end_time"] = parse_time_column(frame["end_time"])
        return frame.dropna(subset=["pod", "service", "node"])

    @staticmethod
    def _wide_metrics(path: Path, entity_column: str, entity_type: str) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if frame.empty:
            return pd.DataFrame(columns=METRIC_COLUMNS)
        required = {"time", entity_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        value_columns = [column for column in frame.columns if column not in required]
        frame = frame.melt(
            id_vars=["time", entity_column],
            value_vars=value_columns,
            var_name="metric",
            value_name="value",
        ).rename(columns={entity_column: "entity"})
        frame["time"] = parse_time_column(frame["time"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["entity_type"] = entity_type
        return frame[METRIC_COLUMNS].dropna(subset=["time", "entity", "value"])

    @staticmethod
    def _pod_metrics(path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if frame.empty:
            return pd.DataFrame(columns=METRIC_COLUMNS)
        required = {"time", "pod", "name", "value"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = frame.rename(columns={"pod": "entity", "name": "metric"})
        frame["time"] = parse_time_column(frame["time"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["entity_type"] = "pod"
        return frame[METRIC_COLUMNS].dropna(subset=["time", "entity", "value"])

    @staticmethod
    def _traces(path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if frame.empty:
            return pd.DataFrame(columns=TRACE_COLUMNS)
        frame = frame.rename(columns={"status_cod": "status_code"})
        for column in TRACE_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        frame["time"] = parse_time_column(frame["time"])
        frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce")
        return frame[TRACE_COLUMNS].dropna(subset=["time", "trace_id", "span_id"])

    @staticmethod
    def _logs(path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if frame.empty:
            return pd.DataFrame(columns=LOG_COLUMNS)
        for column in LOG_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        frame["time"] = parse_time_column(frame["time"])
        frame["message"] = frame["message"].fillna("").astype(str)
        return frame[LOG_COLUMNS].dropna(subset=["time"])


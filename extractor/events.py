import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


MetricKey = Tuple[str, str, str]
TraceKey = Tuple[str, str, str, str]


@dataclass
class RobustStats:
    median: float
    scale: float
    count: int


def robust_stats(values: Iterable[float]) -> Optional[RobustStats]:
    array = np.asarray(list(values), dtype="float64")
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(array))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = max(abs(median) * 1e-6, 1e-6)
    return RobustStats(median=median, scale=scale, count=len(array))


def is_counter(metric: str) -> bool:
    lowered = metric.lower()
    return lowered.endswith("_total") or lowered.endswith("_count")


def metric_feature_groups(frame: pd.DataFrame):
    if frame.empty:
        return
    group_columns = ["entity_type", "entity", "metric"]
    for key, group in frame.groupby(group_columns, dropna=False):
        group = group.sort_values("time")
        values = pd.to_numeric(group["value"], errors="coerce")
        if is_counter(str(key[2])):
            values = values.diff().where(lambda item: item >= 0)
        yield key, values.to_numpy(dtype="float64")


class MetricBaselineBuilder:
    def __init__(self):
        self.values = defaultdict(list)

    def update(self, frame: pd.DataFrame):
        for (scope, entity, metric), values in metric_feature_groups(frame):
            clean = values[np.isfinite(values)].tolist()
            self.values[(str(scope), str(entity), str(metric))].extend(clean)
            self.values[(str(scope), "*", str(metric))].extend(clean)

    def finish(self) -> Dict[MetricKey, RobustStats]:
        return {
            key: stats
            for key, values in self.values.items()
            if (stats := robust_stats(values)) is not None
        }


def _metric_test_values(
    normal: pd.DataFrame, abnormal: pd.DataFrame, key
) -> np.ndarray:
    scope, entity, metric = key
    current = abnormal[
        (abnormal["entity_type"] == scope)
        & (abnormal["entity"] == entity)
        & (abnormal["metric"] == metric)
    ].sort_values("time")
    values = pd.to_numeric(current["value"], errors="coerce")
    if is_counter(str(metric)):
        previous = normal[
            (normal["entity_type"] == scope)
            & (normal["entity"] == entity)
            & (normal["metric"] == metric)
        ].sort_values("time")["value"]
        if not previous.empty:
            values = pd.concat([previous.tail(1), values], ignore_index=True).diff().iloc[1:]
        else:
            values = values.diff()
        values = values.where(values >= 0)
    return values.to_numpy(dtype="float64")


def extract_metric_events(
    normal: pd.DataFrame,
    abnormal: pd.DataFrame,
    baseline: Dict[MetricKey, RobustStats],
    threshold: float = 3.5,
    severe_threshold: float = 6.0,
    min_points: int = 2,
) -> List[List[str]]:
    events = []
    if abnormal.empty:
        return events
    for raw_key in abnormal[["entity_type", "entity", "metric"]].drop_duplicates().itertuples(index=False, name=None):
        scope, entity, metric = map(str, raw_key)
        stats = baseline.get((scope, entity, metric)) or baseline.get((scope, "*", metric))
        if stats is None:
            continue
        values = _metric_test_values(normal, abnormal, raw_key)
        scores = (values - stats.median) / stats.scale
        finite_scores = scores[np.isfinite(scores)]
        if len(finite_scores) == 0:
            continue
        abnormal_scores = finite_scores[np.abs(finite_scores) >= threshold]
        max_index = int(np.argmax(np.abs(finite_scores)))
        max_score = float(finite_scores[max_index])
        if len(abnormal_scores) >= min_points or abs(max_score) >= severe_threshold:
            events.append([entity, metric, "up" if max_score > 0 else "down"])
    return sorted(events)


def derive_trace_calls(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["time", "src", "dst", "operation", "duration", "status_code"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    parents = frame[["trace_id", "span_id", "service"]].dropna().drop_duplicates(
        subset=["trace_id", "span_id"]
    )
    parents = parents.rename(
        columns={"span_id": "parent_span_id", "service": "src"}
    )
    children = frame.merge(
        parents,
        on=["trace_id", "parent_span_id"],
        how="left",
    ).rename(columns={"service": "dst"})
    children = children.dropna(subset=["src", "dst", "operation"])
    children = children[children["src"].astype(str) != children["dst"].astype(str)]
    return children[columns].copy()


def _is_error_status(value) -> bool:
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE", "OK", "SUCCESS", "0", "200", "201", "202", "204"}:
        return False
    try:
        return int(float(text)) >= 400
    except ValueError:
        return "ERROR" in text or "FAIL" in text or "TIMEOUT" in text


def trace_window_features(frame: pd.DataFrame, frequency: str = "30s") -> pd.DataFrame:
    calls = derive_trace_calls(frame)
    columns = ["src", "dst", "operation", "window", "latency", "error_ratio", "call_count"]
    if calls.empty:
        return pd.DataFrame(columns=columns)
    calls["is_error"] = calls["status_code"].map(_is_error_status).astype(float)
    calls["duration"] = pd.to_numeric(calls["duration"], errors="coerce")
    calls["window"] = calls["time"].dt.floor(frequency)
    grouped = calls.groupby(["src", "dst", "operation", "window"], dropna=False)
    result = grouped.agg(
        latency=("duration", lambda values: float(np.nanpercentile(values, 95)) if values.notna().any() else np.nan),
        error_ratio=("is_error", "mean"),
        call_count=("trace_id", "size") if "trace_id" in calls else ("operation", "size"),
    ).reset_index()
    return result[columns]


class TraceBaselineBuilder:
    FEATURES = ("latency", "error_ratio", "call_count")

    def __init__(self, frequency: str = "30s"):
        self.values = defaultdict(list)
        self.frequency = frequency

    def update(self, frame: pd.DataFrame):
        features = trace_window_features(frame, self.frequency)
        for row in features.itertuples(index=False):
            for feature in self.FEATURES:
                value = getattr(row, feature)
                if pd.isna(value):
                    continue
                exact = (str(row.src), str(row.dst), str(row.operation), feature)
                operation_fallback = ("*", "*", str(row.operation), feature)
                global_fallback = ("*", "*", "*", feature)
                self.values[exact].append(float(value))
                self.values[operation_fallback].append(float(value))
                self.values[global_fallback].append(float(value))

    def finish(self) -> Dict[TraceKey, RobustStats]:
        return {
            key: stats
            for key, values in self.values.items()
            if (stats := robust_stats(values)) is not None
        }


def extract_trace_events(
    normal: pd.DataFrame,
    abnormal: pd.DataFrame,
    baseline: Dict[TraceKey, RobustStats],
    threshold: float = 3.5,
    frequency: str = "30s",
) -> List[List[str]]:
    normal_features = trace_window_features(normal, frequency)
    features = trace_window_features(abnormal, frequency)
    events = set()
    event_names = {
        "latency": "latency",
        "error_ratio": "error",
        "call_count": "call_count",
    }
    for row in features.itertuples(index=False):
        for feature, event_name in event_names.items():
            key = (str(row.src), str(row.dst), str(row.operation), feature)
            stats = (
                baseline.get(key)
                or baseline.get(("*", "*", str(row.operation), feature))
                or baseline.get(("*", "*", "*", feature))
            )
            value = getattr(row, feature)
            if stats is None or pd.isna(value):
                continue
            score = (float(value) - stats.median) / stats.scale
            # Low call count and high latency/error ratio are both meaningful.
            if abs(score) >= threshold:
                direction = "up" if score > 0 else "down"
                events.add((str(row.src), str(row.dst), str(row.operation), f"{event_name}_{direction}"))
    # A relation seen during this case's normal window but completely absent
    # from the longer abnormal window is itself a useful network event.
    normal_relations = {
        (str(row.src), str(row.dst), str(row.operation))
        for row in normal_features.itertuples(index=False)
    }
    abnormal_relations = {
        (str(row.src), str(row.dst), str(row.operation))
        for row in features.itertuples(index=False)
    }
    for src, dst, operation in normal_relations - abnormal_relations:
        events.add((src, dst, operation, "call_count_down"))
    return [list(event) for event in sorted(events)]


def normalize_log_message(message: str) -> str:
    text = str(message).strip()
    if not text:
        return text
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(parsed, dict):
        return str(parsed.get("message") or parsed.get("msg") or text)
    return text


class LogTemplateModel:
    ERROR_KEYWORDS = ("error", "fail", "exception", "timeout", "fatal", "panic")

    def __init__(self, config_path: Path):
        config = TemplateMinerConfig()
        config.load(str(config_path))
        config.profiling_enabled = False
        self.miner = TemplateMiner(config=config)
        self.template_counts = defaultdict(int)
        self.rare_template_ids = set()
        self.error_template_ids = set()

    def update(self, frame: pd.DataFrame):
        if frame.empty:
            return
        for message in frame["message"].map(normalize_log_message):
            if not message:
                continue
            result = self.miner.add_log_message(message)
            cluster_id = int(result["cluster_id"])
            self.template_counts[cluster_id] += 1

    def finish(self, rare_quantile: float = 0.2):
        if self.template_counts:
            cutoff = float(np.quantile(list(self.template_counts.values()), rare_quantile))
            self.rare_template_ids = {
                cluster_id
                for cluster_id, count in self.template_counts.items()
                if count <= cutoff
            }
        for cluster in self.miner.drain.clusters:
            template = cluster.get_template().lower()
            if any(keyword in template for keyword in self.ERROR_KEYWORDS):
                self.error_template_ids.add(int(cluster.cluster_id))

    def match(self, message: str) -> int:
        cluster = self.miner.match(normalize_log_message(message))
        return -1 if cluster is None else int(cluster.cluster_id)


def _template_rates(frame: pd.DataFrame, model: LogTemplateModel) -> Dict[Tuple[str, int], float]:
    if frame.empty:
        return {}
    counts = defaultdict(int)
    for row in frame.itertuples(index=False):
        pod = "" if pd.isna(row.pod) else str(row.pod)
        node = "" if pd.isna(row.node) else str(row.node)
        entity = pod or node
        if not entity:
            continue
        counts[(entity, model.match(row.message))] += 1
    duration_minutes = max(
        (frame["time"].max() - frame["time"].min()).total_seconds() / 60.0,
        1.0,
    )
    return {key: count / duration_minutes for key, count in counts.items()}


def extract_log_events(
    normal: pd.DataFrame,
    abnormal: pd.DataFrame,
    model: LogTemplateModel,
    rate_multiplier: float = 3.0,
    rate_delta: float = 0.5,
) -> List[List[str]]:
    normal_rates = _template_rates(normal, model)
    abnormal_rates = _template_rates(abnormal, model)
    events = []
    for (entity, template_id), abnormal_rate in abnormal_rates.items():
        normal_rate = normal_rates.get((entity, template_id), 0.0)
        selected = (
            template_id == -1
            or template_id in model.rare_template_ids
            or template_id in model.error_template_ids
            or abnormal_rate >= max(normal_rate * rate_multiplier, normal_rate + rate_delta)
        )
        if selected:
            events.append([entity, str(template_id)])
    return sorted(events)


def save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


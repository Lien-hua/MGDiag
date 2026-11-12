import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

from extractor.events import (
    LogTemplateModel,
    MetricBaselineBuilder,
    TraceBaselineBuilder,
    extract_log_events,
    extract_metric_events,
    extract_trace_events,
    save_pickle,
)
from extractor.registry import create_adapter


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
    temporary.replace(path)


DEFAULT_SETTINGS = {
    "metric_threshold": 3.5,
    "metric_severe_threshold": 6.0,
    "metric_min_points": 2,
    "trace_threshold": 3.5,
    "trace_window": "30s",
    "log_rare_quantile": 0.2,
    "log_rate_multiplier": 3.0,
    "log_rate_delta": 0.5,
}


def load_settings(input_dir: Path) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    path = input_dir / "extractor_config.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        unknown = sorted(set(overrides) - set(settings))
        if unknown:
            raise ValueError(f"unsupported keys in {path}: {unknown}")
        settings.update(overrides)
    return settings


def fit_extractors(adapter, cases, artifact_dir: Path, settings: dict):
    metric_builder = MetricBaselineBuilder()
    trace_builder = TraceBaselineBuilder(settings["trace_window"])
    log_model = LogTemplateModel(Path(__file__).parent / "drain" / "drain3.ini")
    train_cases = [case for case in cases if case.split == "train"]
    for spec in tqdm(train_cases, desc="Fit extractors", unit="case"):
        data = adapter.load_case(spec)
        metric_builder.update(data.normal_metrics)
        trace_builder.update(data.normal_traces)
        log_model.update(data.normal_logs)
    metric_baseline = metric_builder.finish()
    trace_baseline = trace_builder.finish()
    log_model.finish(settings["log_rare_quantile"])
    save_pickle(artifact_dir / "metric_baseline.pkl", metric_baseline)
    save_pickle(artifact_dir / "trace_baseline.pkl", trace_baseline)
    save_pickle(artifact_dir / "drain.pkl", log_model)
    return metric_baseline, trace_baseline, log_model


def run_pipeline(dataset: str, input_dir: Path, output_dir: Path) -> None:
    adapter = create_adapter(dataset, input_dir)
    cases = adapter.cases()
    settings = load_settings(input_dir)
    metric_baseline, trace_baseline, log_model = fit_extractors(
        adapter, cases, output_dir / "artifacts", settings
    )

    labels = []
    metrics: Dict[str, List[List[str]]] = {}
    traces: Dict[str, List[List[str]]] = {}
    logs: Dict[str, List[List[str]]] = {}
    nodes: Dict[str, List[str]] = {}
    edges: Dict[str, List[List[int]]] = {}
    node_types: Dict[str, Dict[str, str]] = {}

    for spec in tqdm(cases, desc="Extract events", unit="case"):
        data = adapter.load_case(spec)
        case_nodes, case_edges, case_node_types = adapter.build_topology(data)
        missing_roots = [root for root in spec.roots if root not in case_nodes]
        if missing_roots:
            raise ValueError(f"{spec.case_id}: roots absent from graph: {missing_roots}")
        metrics[spec.case_id] = extract_metric_events(
            data.normal_metrics,
            data.abnormal_metrics,
            metric_baseline,
            threshold=settings["metric_threshold"],
            severe_threshold=settings["metric_severe_threshold"],
            min_points=settings["metric_min_points"],
        )
        traces[spec.case_id] = extract_trace_events(
            data.normal_traces,
            data.abnormal_traces,
            trace_baseline,
            threshold=settings["trace_threshold"],
            frequency=settings["trace_window"],
        )
        logs[spec.case_id] = extract_log_events(
            data.normal_logs,
            data.abnormal_logs,
            log_model,
            rate_multiplier=settings["log_rate_multiplier"],
            rate_delta=settings["log_rate_delta"],
        )
        nodes[spec.case_id] = case_nodes
        edges[spec.case_id] = case_edges
        node_types[spec.case_id] = case_node_types
        labels.append(
            {
                "index": spec.case_id,
                "data_type": spec.split,
                "instances_json": json.dumps(spec.roots, ensure_ascii=False),
                "instance_type": spec.root_type,
                "anomaly_type": spec.failure_type,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels).to_csv(output_dir / "label.csv", index=False)
    raw_dir = output_dir / "raw"
    write_json(raw_dir / "metrics.json", metrics)
    write_json(raw_dir / "traces.json", traces)
    write_json(raw_dir / "logs.json", logs)
    write_json(raw_dir / "nodes.json", nodes)
    write_json(raw_dir / "edges.json", edges)
    write_json(raw_dir / "node_types.json", node_types)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw telemetry into MGDiag's dataset-independent schema."
    )
    parser.add_argument("--dataset", required=True, help="registered dataset adapter")
    parser.add_argument("--input", type=Path, help="raw dataset directory")
    parser.add_argument("--output", type=Path, help="processed dataset directory")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input or Path("data") / args.dataset
    output_dir = args.output or Path("data") / args.dataset
    run_pipeline(args.dataset, input_dir, output_dir)


if __name__ == "__main__":
    main()


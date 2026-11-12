from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    split: str
    roots: List[str]
    root_type: str
    failure_type: str
    normal_start: pd.Timestamp
    normal_end: pd.Timestamp
    abnormal_start: pd.Timestamp
    abnormal_end: pd.Timestamp
    path: Path


@dataclass
class CaseData:
    spec: CaseSpec
    normal_metrics: pd.DataFrame
    abnormal_metrics: pd.DataFrame
    normal_traces: pd.DataFrame
    abnormal_traces: pd.DataFrame
    normal_logs: pd.DataFrame
    abnormal_logs: pd.DataFrame


class DatasetAdapter(ABC):
    """Convert one raw dataset into the canonical extractor schema."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @abstractmethod
    def cases(self) -> List[CaseSpec]:
        raise NotImplementedError

    @abstractmethod
    def load_case(self, spec: CaseSpec) -> CaseData:
        raise NotImplementedError

    @abstractmethod
    def build_topology(
        self, data: CaseData
    ) -> Tuple[List[str], List[List[int]], Dict[str, str]]:
        """Return node names, local-index edges, and node name -> granularity."""
        raise NotImplementedError


"""稳定且原子化的批量仿真结果存储接口。"""

from .schema import CanonicalResultMetadata, validate_canonical_summary
from .results import read_trace_table

__all__ = [
    "CanonicalResultMetadata",
    "read_trace_table",
    "validate_canonical_summary",
]

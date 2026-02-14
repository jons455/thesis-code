"""Registry for metric key mappings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricRegistry:
    """Map semantic metric names to task-specific keys."""

    mapping: dict[str, str] = field(default_factory=dict)

    def register(self, metric: str, key: str) -> None:
        self.mapping[metric] = key

    def resolve(self, metric: str) -> str | None:
        return self.mapping.get(metric)

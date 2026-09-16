"""The immutable configuration for one point in an experiment sweep."""

from dataclasses import dataclass, fields
from typing import Mapping

import results


@dataclass(frozen=True)
class ExperimentConfig:
    """Every value one experiment receives after sweep expansion."""

    task: str
    mas_type: str
    mas_memory: str
    reasoning: str
    model: str
    max_trials: int | None
    max_tasks: int | None
    successful_topk: int
    failed_topk: int
    insights_topk: int
    threshold: float
    use_projector: bool
    use_validator: bool
    hop: int
    intrinsic_cross_task: bool
    max_tokens: int
    max_tokens_ceiling: int
    thinking_token_budget: int | None
    temperature: float
    request_timeout: float
    log_responses: bool
    wikipedia_attempts: int
    wikipedia_retry_seconds: float
    unreachable_search_limit: int
    seed: int
    num_workers: int
    resume: bool
    db_dir: str
    overall_results_filename: str
    failed_tasks_filename: str
    failed_experiments_filename: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ExperimentConfig":
        """Build a config, refusing parser/dataclass schema drift."""
        expected = {field.name for field in fields(cls)}
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise TypeError(f"ExperimentConfig fields do not match parser ({'; '.join(details)})")
        return cls(**dict(values))

    def identity(self) -> dict[str, object]:
        """Fields that identify this experiment in result rows."""
        return {column: getattr(self, column) for column in results.IDENTITY_COLUMNS}

    def experiment_key(self) -> tuple[str, ...]:
        """The `--resume` key, derived from the result schema."""
        return tuple(str(getattr(self, column)) for column in results.KEY_COLUMNS)

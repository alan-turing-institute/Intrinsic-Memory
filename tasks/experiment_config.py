"""The immutable configuration for one point in an experiment sweep."""

from dataclasses import dataclass, fields
from typing import Mapping


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
        return {
            'model': self.model,
            'task': self.task,
            'mas_type': self.mas_type,
            'mas_memory': self.mas_memory,
            'use_validator': self.use_validator,
            'intrinsic_cross_task': self.intrinsic_cross_task,
        }

    def experiment_key(self) -> tuple[str, ...]:
        """The `--resume` key, using the same values written to result rows."""
        return tuple(
            str(value)
            for value in (
                self.model,
                self.task,
                self.mas_type,
                self.mas_memory,
                self.use_validator,
                self.intrinsic_cross_task,
                self.seed,
            )
        )

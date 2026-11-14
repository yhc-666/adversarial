from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class EvolutionConfig:
    model: str
    verifier_model: str
    max_rounds: int = 7
    reference_count: ClassVar[int] = 1
    attack_max_attempts: ClassVar[int] = 3
    repair_max_attempts: ClassVar[int] = 3
    format_max_attempts: ClassVar[int] = 2
    no_update_patience: ClassVar[int] = 3

    def __post_init__(self) -> None:
        for name in ("model", "verifier_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.max_rounds) is not int or self.max_rounds < 0:
            raise ValueError("max_rounds must be a non-negative integer")

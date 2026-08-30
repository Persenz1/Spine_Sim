"""Structured error categories; physical infeasibility is not an exception bucket."""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    INPUT_CONFIGURATION = "input_configuration"
    MODEL_UNCLOSED = "model_unclosed"
    EXECUTION = "execution"


class SpineSimError(Exception):
    category = ErrorCategory.EXECUTION

    def as_dict(self) -> dict[str, str]:
        return {
            "category": self.category.value,
            "type": type(self).__name__,
            "message": str(self),
        }


class ConfigurationError(SpineSimError):
    category = ErrorCategory.INPUT_CONFIGURATION


class ModelUnclosedError(SpineSimError):
    category = ErrorCategory.MODEL_UNCLOSED


def classify_exception(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, SpineSimError):
        return exc.as_dict()
    return {
        "category": ErrorCategory.EXECUTION.value,
        "type": type(exc).__name__,
        "message": str(exc),
    }

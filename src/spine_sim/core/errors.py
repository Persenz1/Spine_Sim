"""Structured error categories; physical infeasibility is not an exception bucket."""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    INPUT_CONFIGURATION = "input_configuration"
    MODEL_UNCLOSED = "model_unclosed"
    NUMERICAL_NONCONVERGENCE = "numerical_nonconvergence"
    EXECUTION = "execution"
    USER_CANCELLED = "user_cancelled"


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


class NumericalConvergenceError(SpineSimError):
    category = ErrorCategory.NUMERICAL_NONCONVERGENCE


class ExecutionError(SpineSimError):
    category = ErrorCategory.EXECUTION


class UserCancelledError(SpineSimError):
    category = ErrorCategory.USER_CANCELLED


def classify_exception(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, SpineSimError):
        return exc.as_dict()
    return {
        "category": ErrorCategory.EXECUTION.value,
        "type": type(exc).__name__,
        "message": str(exc),
    }

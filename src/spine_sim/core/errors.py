"""结构化异常分类。

这里只表示输入错误、模型未闭合或程序执行失败；正常的物理不可行由结果状态表达，
不应抛成异常。
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    """持久化到运行结果中的顶层错误类别。"""

    INPUT_CONFIGURATION = "input_configuration"
    MODEL_UNCLOSED = "model_unclosed"
    EXECUTION = "execution"


class SpineSimError(Exception):
    """可被 runner 直接结构化记录的仿真异常基类。"""

    category = ErrorCategory.EXECUTION

    def as_dict(self) -> dict[str, str]:
        """输出稳定的类别、异常类型和消息。"""

        return {
            "category": self.category.value,
            "type": type(self).__name__,
            "message": str(self),
        }


class ConfigurationError(SpineSimError):
    """输入或配置不满足已声明契约。"""

    category = ErrorCategory.INPUT_CONFIGURATION


class ModelUnclosedError(SpineSimError):
    """所需物理参数或模型范围尚未闭合。"""

    category = ErrorCategory.MODEL_UNCLOSED


def classify_exception(exc: BaseException) -> dict[str, str]:
    """保留已知异常类别，其余异常统一标为执行错误。"""

    if isinstance(exc, SpineSimError):
        return exc.as_dict()
    return {
        "category": ErrorCategory.EXECUTION.value,
        "type": type(exc).__name__,
        "message": str(exc),
    }

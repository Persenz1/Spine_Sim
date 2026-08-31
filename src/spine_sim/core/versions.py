"""参与 case/result identity 的语义版本常量。

这些值描述数据和求解语义，而不是普通发布版本；一旦兼容含义改变就必须升级，
从而避免新旧结果共享同一个 identity。
"""

PROJECT_SCHEMA_VERSION = "2"
MODEL_SCHEMA_VERSION = "canonical-single-array-1"
RESULT_SCHEMA_VERSION = "canonical-result-2"
SOLVER_SEMANTICS_VERSION = "single-array-event-v1"
GEOMETRY_SCHEMA_VERSION = "contact-candidate-2"
PARAMETER_REGISTRY_VERSION = "canonical-parameters-1"

# 结果中使用的模型层级标签，用于区分单刺和刚性背板阵列输出契约。
SINGLE_SPINE_MODEL_LEVEL = "single_spine_quasistatic"
ARRAY_MODEL_LEVEL = "array_rigid_backplate_event"

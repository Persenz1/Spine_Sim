"""M1 地形层的结构化错误类型。"""

from __future__ import annotations

from spine_sim.core.errors import ConfigurationError, ModelUnclosedError


class TerrainConfigurationError(ConfigurationError):
    """地形配方、区域或缓存请求内部不一致。"""


class GeometryOutOfDomainError(ModelUnclosedError):
    """文件高度图或 track 查询离开有效来源域。"""

    code = "geometry_out_of_domain"


class RodCollisionModelUnclosedError(ModelUnclosedError):
    """杆体/圆柱碰撞参数不足以作出确定物理判定。"""

    code = "model_unclosed_rod_collision"

"""M1-specific structured errors."""

from __future__ import annotations

from spine_sim.core.errors import ConfigurationError, ModelUnclosedError


class TerrainConfigurationError(ConfigurationError):
    """A terrain recipe, region or cache request is inconsistent."""


class GeometryOutOfDomainError(ModelUnclosedError):
    """A requested file-heightmap or track query leaves its valid source domain."""

    code = "geometry_out_of_domain"


class RodCollisionModelUnclosedError(ModelUnclosedError):
    """Rod/cylinder collision parameters are not sufficient for a physical decision."""

    code = "model_unclosed_rod_collision"

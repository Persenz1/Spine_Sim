"""M2-specific configuration and geometry failures."""

from spine_sim.core.errors import ConfigurationError


class ContactConfigurationError(ConfigurationError):
    """Raised when a single-spine model is not physically closed."""


class ContactGeometryError(RuntimeError):
    """Raised when a requested pose cannot query valid M1 geometry."""

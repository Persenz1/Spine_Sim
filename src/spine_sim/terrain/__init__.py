"""M1 terrain recipes, local caches and finite-tip track geometry."""

from .analytic import evaluate_analytic
from .envelope import (
    RodClearanceResult,
    SphereEnvelope2D,
    check_rod_clearance,
    compute_sphere_envelope_2d,
    compute_track_geometry,
    forward_cap_gate,
)
from .heightmap import (
    FileHeightMapSource,
    register_heightmap_source,
    sample_file_heightmap,
)
from .library import TerrainLibrary
from .models import (
    M1_MODULE_VERSION,
    CampaignDesignSpace,
    CampaignRegionReport,
    RegionSpec,
    TerrainRecipe,
    TrackGeometry,
    compute_campaign_region,
)
from .plotting import (
    GrooveSelection,
    SpherePlacement,
    TerrainPatch,
    extract_centered_patch,
    place_sphere_on_patch,
    render_terrain_views,
    select_groove_center,
)
from .random_field import generate_defined_geometry
from .suite import generate_terrain_suite

__all__ = [
    "M1_MODULE_VERSION",
    "CampaignDesignSpace",
    "CampaignRegionReport",
    "FileHeightMapSource",
    "GrooveSelection",
    "RegionSpec",
    "RodClearanceResult",
    "SphereEnvelope2D",
    "SpherePlacement",
    "TerrainLibrary",
    "TerrainPatch",
    "TerrainRecipe",
    "TrackGeometry",
    "check_rod_clearance",
    "compute_campaign_region",
    "compute_sphere_envelope_2d",
    "compute_track_geometry",
    "evaluate_analytic",
    "extract_centered_patch",
    "forward_cap_gate",
    "generate_defined_geometry",
    "generate_terrain_suite",
    "place_sphere_on_patch",
    "register_heightmap_source",
    "sample_file_heightmap",
    "render_terrain_views",
    "select_groove_center",
]

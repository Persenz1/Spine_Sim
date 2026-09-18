"""M1 地形配方、本地缓存及有限半径尖端的轨迹几何接口。"""

from .analytic import evaluate_analytic
from .api import (
    Terrain,
    generate_terrain,
    refine_material_terrain_same_realization,
    load_terrain,
    register_terrain,
    save_terrain,
)
from .descriptors import compute_descriptors
from .envelope import (
    RodClearanceResult,
    SphereEnvelope2D,
    check_rod_clearance,
    check_segmented_tip_rod_clearance,
    compute_sphere_envelope_2d,
    compute_track_geometry,
    forward_cap_gate,
)
from .measured import (
    FileHeightMapSource,
    register_heightmap_source,
    sample_file_heightmap,
)
from .library import TerrainLibrary
from .models import (
    ENVELOPE_ALGORITHM_VERSION,
    M1_MODULE_VERSION,
    MATERIAL_TERRAIN_VERSION,
    TRACK_SCHEMA_VERSION,
    CampaignDesignSpace,
    CampaignRegionReport,
    RegionSpec,
    TerrainRecipe,
    TrackGeometry,
    compute_campaign_region,
)
from .profiles import available_profiles, load_material_profile
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
from .validation import (
    compare_topographies,
    render_comparison,
    summarize_seed_ensemble,
)

__all__ = [
    "M1_MODULE_VERSION",
    "MATERIAL_TERRAIN_VERSION",
    "ENVELOPE_ALGORITHM_VERSION",
    "TRACK_SCHEMA_VERSION",
    "CampaignDesignSpace",
    "CampaignRegionReport",
    "FileHeightMapSource",
    "GrooveSelection",
    "RegionSpec",
    "RodClearanceResult",
    "SphereEnvelope2D",
    "SpherePlacement",
    "TerrainLibrary",
    "Terrain",
    "TerrainPatch",
    "TerrainRecipe",
    "TrackGeometry",
    "check_rod_clearance",
    "check_segmented_tip_rod_clearance",
    "compute_campaign_region",
    "compute_descriptors",
    "compute_sphere_envelope_2d",
    "compute_track_geometry",
    "evaluate_analytic",
    "extract_centered_patch",
    "forward_cap_gate",
    "generate_defined_geometry",
    "generate_terrain",
    "refine_material_terrain_same_realization",
    "generate_terrain_suite",
    "place_sphere_on_patch",
    "available_profiles",
    "compare_topographies",
    "load_material_profile",
    "load_terrain",
    "register_terrain",
    "register_heightmap_source",
    "sample_file_heightmap",
    "render_terrain_views",
    "render_comparison",
    "save_terrain",
    "select_groove_center",
    "summarize_seed_ensemble",
]

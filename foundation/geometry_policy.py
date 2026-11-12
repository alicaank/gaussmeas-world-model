"""Version single-source renders without invalidating multi-source caches."""
from pathlib import Path

SINGLE_SOURCE_POLICY = 'duplicate_identical_view_for_resplat.v1'
SINGLE_SOURCE_DIRECTORY = 'geometry_single_resplat_v1'


def geometry_path(root, recipe, suffix):
    directory = SINGLE_SOURCE_DIRECTORY if len(recipe['source_frame_ids']) == 1 else 'geometry'
    return Path(root) / directory / (recipe['recipe_id'] + suffix)


def validate_geometry_policy(metadata, recipe):
    if len(recipe['source_frame_ids']) == 1:
        if metadata.get('provenance', {}).get('single_source_policy') != SINGLE_SOURCE_POLICY:
            raise ValueError('single-source cache was not produced by duplicate-view ReSplat')

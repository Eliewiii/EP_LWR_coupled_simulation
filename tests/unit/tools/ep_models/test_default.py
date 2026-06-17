"""Unit tests validating production-grade baseline presets, enum contracts, and the master registry
builder.
"""

import pytest
from lwrepcoupling.tools.ep_models.default import (
    DefaultConstructionNames,
    get_default_base_registry,
    get_default_building_registry,
    get_default_external_wall_construction_registry,
    get_default_materials_registry,
    get_default_timestep_registry,
)
from pyenergyplus.model.model import SolarDistribution, Terrain


def test_construction_names_enum_values():
    """
    Validates that explicit construction string accessors match expected EnergyPlus schema
    contracts.
    """
    assert DefaultConstructionNames.WALL.value == "Default_Exterior_Wall_Construction"
    assert DefaultConstructionNames.ROOF.value == "Default_Roof_Construction"
    assert DefaultConstructionNames.FLOOR.value == "Default_Ground_Floor_Construction"
    assert DefaultConstructionNames.WINDOW.value == "Default_Double_Glazing_Construction"


def test_building_registry_rigid_syntax_keywords():
    """
    Ensures critical EnergyPlus physics and boundary enums are explicitly populated to prevent
    engine syntax faults.
    """
    building_map = get_default_building_registry("Test_Bldg")
    assert "Test_Bldg" in building_map

    bldg_obj = building_map["Test_Bldg"]
    # Rigid verification: EnergyPlus will fail if these exact schema enums are missing
    # or unvalidated
    assert bldg_obj.terrain == Terrain.urban
    assert bldg_obj.solar_distribution == SolarDistribution.full_exterior_with_reflections
    assert bldg_obj.maximum_number_of_warmup_days == 25


def test_timestep_registry_bounds():
    """Validates discrete simulation time-step registration rules."""
    timestep_map = get_default_timestep_registry(timesteps_per_hour=6)
    assert "Timestep 1" in timestep_map
    assert timestep_map["Timestep 1"].number_of_timesteps_per_hour == 6


def test_wall_construction_layer_validation_raises_on_missing_materials():
    """
    Asserts that generating an external wall construction with an unregistered material sequence
    triggers a strict ValueError.
    """
    mock_material_pool = (
        get_default_materials_registry()
    )  # Missing other critical envelope materials
    mock_material_pool.pop(
        "Common_Brick", None
    )  # Explicitly remove a required material to simulate a missing layer

    with pytest.raises(
        ValueError, match="Material 'Common_Brick' assigned to wall construction does not exist"
    ):
        get_default_external_wall_construction_registry(
            material_pool=mock_material_pool, layer_outside="Common_Brick"
        )


def test_master_base_registry_complete_composition():
    """
    Guarantees the comprehensive base registry compiles all required foundational sub-dictionaries
    cleanly.
    """
    base_registry = get_default_base_registry()

    # Structural cross-check against the SimulationBaseRegistry layout definition
    assert len(base_registry.building) == 1
    assert len(base_registry.global_geometry_rules) == 1
    assert len(base_registry.material) >= 5
    assert len(base_registry.construction) >= 4

    # Assert that required standard materials exist within the pool contract
    assert "Heavy_Concrete" in base_registry.material
    assert "Wall_Roof_Insulation" in base_registry.material

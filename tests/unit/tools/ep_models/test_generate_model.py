"""
Unit tests validating the core programmatic compilation backend engine and structural object
stitching rules.
"""

from unittest.mock import MagicMock, patch

import pytest
from lwrepcoupling.tools.ep_models.default import (
    get_default_base_registry,
    get_single_zone_box_geometry,
)
from lwrepcoupling.tools.ep_models.generate_model import (
    _get_installed_ep_version,
    compile_energyplus_model,
)
from pyenergyplus.model import EnergyPlusModel


@patch("lwrepcoupling.tools.ep_models.generate_model.EnergyPlusAPI")
def test_get_installed_ep_version_string_parsing(mock_api_class):
    """
    Verifies that the engine extracts system major/minor attributes and formats a correct
    semantic string.
    """
    # Setup mock functional library interface structure matching EnergyPlusAPI
    mock_api_instance = MagicMock()
    mock_api_instance.functional.ep_version.return_value.ep_version_major = 24
    mock_api_instance.functional.ep_version.return_value.ep_version_minor = 1
    mock_api_class.return_value = mock_api_instance

    version_str = _get_installed_ep_version()
    assert version_str == "24.1"


@patch("lwrepcoupling.tools.ep_models.generate_model.EnergyPlusAPI")
def test_get_installed_ep_version_api_failure_raises_runtime_error(mock_api_class):
    """
    Guarantees that if the underlying native API crashes or cannot be bound, a descriptive
    RuntimeError is raised.
    """
    mock_api_class.side_effect = Exception("Shared library footprint missing.")

    with pytest.raises(
        RuntimeError, match="Failed to extract installed EnergyPlus version via API bindings"
    ):
        _get_installed_ep_version()


@patch("lwrepcoupling.tools.ep_models.generate_model._get_installed_ep_version")
def test_compile_energyplus_model_stitching_and_injection(mock_version_func):
    """
    Validates that the compiler cleanly combines geometry records, material states, and
    global simulation properties.
    """
    mock_version_func.return_value = "23.2"

    # 1. Fetch default presets and raw geometric layout components
    base_registry = get_default_base_registry()
    geo_data = get_single_zone_box_geometry(zone_name="isolated_box", width=8, length=8, height=3)

    # 2. Execute compilation handoff
    compiled_model = compile_energyplus_model(
        base_registry=base_registry,
        building_zone_registry=geo_data["Zone"],
        building_surface_registry=geo_data["BuildingSurfaceDetailed"],
    )

    # 3. Validation assertions against strict EnergyPlusModel property contracts
    assert isinstance(compiled_model, EnergyPlusModel)

    # Assert system version tracking properties injected cleanly
    assert compiled_model.version and compiled_model.version["Version"].version_identifier == "23.2"

    # Assert structural geometry registries translated accurately
    assert compiled_model.zone and "isolated_box" in compiled_model.zone
    assert (
        compiled_model.building_surface_detailed
        and "isolated_box_roof" in compiled_model.building_surface_detailed
    )
    assert compiled_model.building_surface_detailed["isolated_box_roof"].zone_name == "isolated_box"

    # Assert simulation parameters safely transferred from base registry
    assert "Main_Building" in compiled_model.building
    assert compiled_model.run_period and "RunPeriod 1" in compiled_model.run_period
    assert compiled_model.material and "Heavy_Concrete" in compiled_model.material
    assert (
        compiled_model.construction
        and "Default_Exterior_Wall_Construction" in compiled_model.construction
    )

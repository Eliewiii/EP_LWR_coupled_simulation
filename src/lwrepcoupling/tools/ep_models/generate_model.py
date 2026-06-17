"""
Core module for programmatically compiling structured object registries into an EnergyPlusModel.
"""

from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.model import EnergyPlusModel
from pyenergyplus.model.model import (
    BuildingSurfaceDetailed,
    Version,
    Zone,
)

from ..schemas import SimulationBaseRegistry


def _get_installed_ep_version() -> str:
    """Return the currently installed EnergyPlus major/minor version.

    The returned string is formatted as ``<major>.<minor>`` and is suitable for
    use as the version identifier in a pyenergyplus ``Version`` object.

    Raises:
        RuntimeError: If the EnergyPlus API fails to initialize or query the version binaries.
    """
    try:
        api = EnergyPlusAPI()
        version_info = api.functional.ep_version()
        return f"{version_info.ep_version_major}.{version_info.ep_version_minor}"
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract installed EnergyPlus version via API bindings. "
            f"Verify your environment installation paths. Error details: {e}"
        ) from e


def compile_energyplus_model(
    base_registry: SimulationBaseRegistry,
    building_zone_registry: dict[str, Zone],
    building_surface_registry: dict[str, BuildingSurfaceDetailed],
) -> EnergyPlusModel:
    """Constructs a new EnergyPlusModel by combining structural geometries with a base registry.

    Args:
        base_registry: A typed container mapping baseline presets (materials, controls, etc.).
        building_zone_registry: Generated Zone object map for the specific case layout.
        building_surface_registry: Generated BuildingSurfaceDetailed object map for the case layout.

    Returns:
        A fully populated EnergyPlusModel instance ready for execution or testing assertions.
    """
    model = EnergyPlusModel.model_construct()

    # 1. Inject internal system version tracking properties safely
    version_id = _get_installed_ep_version()
    model.version = {"Version": Version(version_identifier=version_id)}

    # 2. Extract and assign foundational physics properties from the base container
    model.building = dict(base_registry.building)
    model.global_geometry_rules = dict(base_registry.global_geometry_rules)
    model.timestep = dict(base_registry.timestep)
    model.simulation_control = dict(base_registry.simulation_control)
    model.output_variable = dict(base_registry.output_variable)
    model.output_sq_lite = dict(base_registry.output_sq_lite)
    model.run_period = dict(base_registry.run_period)
    model.material = dict(base_registry.material)
    model.construction = dict(base_registry.construction)

    # 3. Inject the explicit case study geometry arguments
    model.zone = dict(building_zone_registry)
    model.building_surface_detailed = dict(building_surface_registry)

    return model

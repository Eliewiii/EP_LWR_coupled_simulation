"""
Main file to test the LWR-EP coupling simulation manager without pytest as it allows print for EnergyPlus to
detect potential errors and problems, for development purposes only
"""
import os

from src.lwrepcoupling import EpLwrSimulationManager
from src.lwrepcoupling.ep_simulation_instance_with_shared_memory import EpSimulationInstance
# Path to input data
test_data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
# Inputs
path_output_dir = os.path.join(test_data_dir, "simulation_dir")
path_epw = os.path.join(test_data_dir, "weather_sample.epw")
path_energyplus_dir = r"C:\EnergyPlusV23-2-0"
path_idf = os.path.join(test_data_dir, "idfs", "in_0.idf")
# path_idf_1 = os.path.join(test_data_dir, "idfs", "in_1.idf")


building_id_list = ["example_0", "example_1"]
outdoor_surface_name_list = ["room_1_bf4c06dc..Face0", "room_1_bf4c06dc..Face1"]  # Check the names
outdoor_surface_surrounding_surface_vf_dict = {"room_1_bf4c06dc..Face0": 0.5, "room_1_bf4c06dc..Face1": 0.5}
outdoor_surface_sky_vf_dict = {"room_1_bf4c06dc..Face0": 0.1, "room_1_bf4c06dc..Face1": 0.1}
outdoor_surface_ground_vf_dict = {"room_1_bf4c06dc..Face0": 0.2, "room_1_bf4c06dc..Face1": 0.2}


def main():
    """
    """
    # Init the object
    ep_lwr_simulation_manager = EpLwrSimulationManager(path_output_dir, path_epw, path_energyplus_dir)
    # Add the buildings
    for building_id in building_id_list:
        # same idf, and surface details for all buildings
        ep_lwr_simulation_manager.add_building(building_id, path_idf, outdoor_surface_name_list,
                                               outdoor_surface_surrounding_surface_vf_dict,
                                               outdoor_surface_sky_vf_dict,
                                               outdoor_surface_ground_vf_dict)
        path_pkl = ep_lwr_simulation_manager._ep_simulation_instance_dict[building_id].to_pkl(path_folder=path_output_dir)
        ep_sim_obj = EpSimulationInstance.from_pkl(path_pkl)
    # Run the simulation
    # return ep_lwr_simulation_manager.run_lwr_coupled_simulation()


if __name__ == "__main__":
    result = main()

    print("done")

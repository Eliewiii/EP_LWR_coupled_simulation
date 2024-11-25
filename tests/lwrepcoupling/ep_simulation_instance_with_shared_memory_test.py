"""

"""

import os

from src.lwrepcoupling import EpSimulationInstance



# Path to input data
test_data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
# Inputs
path_output_dir = os.path.join(test_data_dir, "simulation_dir")
path_epw = os.path.join(test_data_dir, "weather_sample.epw")
path_energyplus_dir = r"C:\EnergyPlusV23-2-0"
path_idf_0 = os.path.join(test_data_dir, "idfs", "in_0.idf")
path_idf_1 = os.path.join(test_data_dir, "idfs", "in_1.idf")
class TestEpSimulationInstanceInit:

    def test_adjust_idf(self):
        identifier = "example_0"
        ep_simulation_instance = EpSimulationInstance(identifier=identifier,
                                                        path_idf=path_idf_0,
                                                        path_output_dir=os.path.join(path_output_dir, f"building_{identifier}"),
                                                        path_energyplus_dir=path_energyplus_dir,
                                                        simulation_index=0)
        outdoor_surface_name_list = ["room_1_bf4c06dc..Face0", "room_1_bf4c06dc..Face1"]
        for outdoor_surface_name in outdoor_surface_name_list:
            ep_simulation_instance.outdoor_surface_name_list.append(
                outdoor_surface_name)
            ep_simulation_instance.outdoor_surface_surrounding_surface_vf_dict[outdoor_surface_name] = 0.5
            ep_simulation_instance.outdoor_surface_sky_vf_dict[outdoor_surface_name] = 0.1
            ep_simulation_instance.outdoor_surface_ground_vf_dict[outdoor_surface_name] = 0.2
        ep_simulation_instance.adjust_idf()




"""

"""
import os

from src.lwrepcoupling import EpLwrSimulationManager

# Path to input data
test_data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
# Inputs
path_output_dir = os.path.join(test_data_dir, "simulation_dir")
path_epw = os.path.join(test_data_dir, "weather_sample.epw")
path_energyplus_dir = r"C:\EnergyPlusV23-2-0"
path_idf_0 = os.path.join(test_data_dir, "idfs", "in_0.idf")
path_idf_1 = os.path.join(test_data_dir, "idfs", "in_1.idf")


class TestEpLwrSimulationManagerInit:

    def test_init(self):
        ep_lwr_simulation_manager = EpLwrSimulationManager(path_output_dir, path_epw, path_energyplus_dir)
        assert ep_lwr_simulation_manager._path_output_dir == path_output_dir
        assert ep_lwr_simulation_manager._path_epw == path_epw
        assert ep_lwr_simulation_manager._path_energyplus_dir == path_energyplus_dir
        assert ep_lwr_simulation_manager.num_building == 0
        assert ep_lwr_simulation_manager.num_outdoor_surfaces == 0

    def test_add_building(self):

        ep_lwr_simulation_manager = EpLwrSimulationManager(path_output_dir, path_epw, path_energyplus_dir)

        path_idf = path_idf_0
        # First building
        building_id = "example_0"
        ep_lwr_simulation_manager.add_building(building_id, path_idf)
        # --- For testing purposes --- #
        # add outdoor surface
        outdoor_surface_name = "room_1_bf4c06dc..Face0"
        ep_lwr_simulation_manager._ep_simulation_instance_dict[building_id].outdoor_surface_name_list.append(
            outdoor_surface_name)
        # --- -------------------- --- #
        assert ep_lwr_simulation_manager.num_building == 1
        assert ep_lwr_simulation_manager.num_outdoor_surfaces == 1
        assert ep_lwr_simulation_manager._building_id_list == [building_id]
        assert ep_lwr_simulation_manager._ep_simulation_instance_dict[building_id]._path_simulated_idf == None
        assert ep_lwr_simulation_manager._ep_simulation_instance_dict[
                   building_id]._path_output_dir == os.path.join(path_output_dir, "building_example_0")
        assert ep_lwr_simulation_manager._ep_simulation_instance_dict[building_id]._simulation_index == 0

        # Second building
        path_idf = path_idf_1
        building_id = "example_1"
        ep_lwr_simulation_manager.add_building(building_id, path_idf)
        # --- For testing purposes --- #
        # add outdoor surface
        outdoor_surface_name = "room_1_bf4c06dc..Face0"
        ep_lwr_simulation_manager._ep_simulation_instance_dict[building_id].outdoor_surface_name_list.append(
            outdoor_surface_name)
        # --- -------------------- --- #
        assert ep_lwr_simulation_manager.num_building == 2
        assert ep_lwr_simulation_manager.num_outdoor_surfaces == 2
        assert ep_lwr_simulation_manager._building_id_list == ["example_0", "example_1"]


    def test_adjust_building_idf(self):

        ep_lwr_simulation_manager = EpLwrSimulationManager(path_output_dir, path_epw, path_energyplus_dir)

        path_idf = path_idf_0
        # First building
        building_id = "example_0"
        ep_lwr_simulation_manager.add_building(building_id, path_idf)
        # --- For testing purposes --- #
        # add outdoor surface
        outdoor_surface_name = "room_1_bf4c06dc..Face0"
        ep_lwr_simulation_manager._ep_simulation_instance_dict[building_id].outdoor_surface_name_list.append(
            outdoor_surface_name)
        # --- -------------------- --- #

        # Second building
        path_idf = path_idf_1
        building_id = "example_1"
        ep_lwr_simulation_manager.add_building(building_id, path_idf)
        # --- For testing purposes --- #
        # add outdoor surface
        outdoor_surface_name = "room_1_bf4c06dc..Face0"
        ep_lwr_simulation_manager._ep_simulation_instance_dict[building_id].outdoor_surface_name_list.append(
            outdoor_surface_name)
        # --- -------------------- --- #
        assert ep_lwr_simulation_manager.num_building == 2
        assert ep_lwr_simulation_manager.num_outdoor_surfaces == 2
        assert ep_lwr_simulation_manager._building_id_list == ["example_0", "example_1"]



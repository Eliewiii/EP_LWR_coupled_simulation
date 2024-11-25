"""
Test the synchronization of the simulation with the API
"""


from src.lwrepcoupling.ep_coupled_simulation_manager import EpLwrSimulationManager

# Inputs
path_energyplus_dir = r"C:\EnergyPlusV23-2-0"

path_idf = r"C:\Users\elie-medioni\OneDrive\OneDrive - Technion\BUA\LWR\pyenergyplus\sample_cube_buildings\IDFs\cube_1_with_surrounding_surface\openstudio\run\in.idf"
building_id_list= ["cube_1_0","cube_1_1"]
path_output = r"C:\Users\elie-medioni\OneDrive\OneDrive - Technion\BUA\LWR\pyenergyplus\sim_folder"

oudoor_surface_name_list = ["room_1_bf4c06dc..Face0"]
schedule_name = "Test_surrounding_surface_temp"


def main():


    # Initialize the simulation manager
    simulation_manager = EpLwrSimulationManager()
    # Init the buildings
    simulation_manager.add_building(building_id_list[0], path_idf, path_energyplus_dir, oudoor_surface_name_list, vf_matrices=None)






if __name__ == "__main__":
    main()
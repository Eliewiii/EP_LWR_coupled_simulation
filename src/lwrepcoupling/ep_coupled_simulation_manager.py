"""
Class to manage the couped long-wave radiation (LWR) simulation with EnergyPlus among multiple buildings.
"""
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, Manager

from .ep_simulation_instance_with_shared_memory import EpSimulationInstance


class EpLwrSimulationManager:
    def __init__(self):
        # Simulation
        self._building_id_list = []
        self._ep_simulation_instance_dict = []

        # Input variables
        self._path_output_dir = None
        self._path_epw = None

        #

    # -----------------------------------------------------#
    # --------------------- Properties --------------------#
    # -----------------------------------------------------#

    @property
    def num_building(self):
        return len(self._building_id_list)

    @property
    def num_outdoor_surfaces(self):
        return sum([len(ep_simulation_instance.outdoor_surface_name_list) for ep_simulation_instance in
                    self._ep_simulation_instance_dict])

    def add_building(self, building_id, path_idf, path_energyplus_dir, oudoor_surface_name_list, vf_matrices):
        """
        Add a building to the simulation manager.

        :param building_id: The building ID.
        :param path_idf: The path to the IDF file for the building.
        """
        # Create an EnergyPlus simulation instance for the building
        ep_simulation_instance = EpSimulationInstance(
            path_idf=path_idf,
            path_epw=self._path_epw,
            path_output_dir=os.path.join(self._path_output_dir, f"building_{building_id}"),
            path_energyplus_dir=path_energyplus_dir,
            simulation_index=self.num_building
        )
        ep_simulation_instance.set_outdoor_surfaces_and_view_factors(oudoor_surface_name_list, vf_matrices,
                                                                     manager_num_outdoor_surfaces=self.num_outdoor_surfaces)
        self._building_id_list.append(building_id)
        self._ep_simulation_instance_dict.append(ep_simulation_instance)

    def set_outdoor_surfaces_and_view_factors(self):
        """

        """
        # todo : implement this method

    def adjust_buildings_idf(self):
        """

        :return:
        """
        for ep_simulation_instance in self._ep_simulation_instance_dict.values():
            ep_simulation_instance.adjust_idf()

    def run_lwr_coupled_simulation(self):
        """

        :return:
        """

        # todo : identify the size of array_shape according to the number of outdoor surfaces of the buildings
        # need to be in the proper format
        shared_memory_array_size = self.num_outdoor_surfaces * np.float64().itemsize

        # Run the simulation under a Manager context to share memory, locks, and barriers
        with Manager() as manager:

            # Initialize a lock to limit writing access to shared memory
            shared_memory_lock = manager.Lock()
            # Initialize a barrier to synchronize processes, when called with .wait() all processes will wait until all
            # processes have reached the barrier
            synch_point_barrier = manager.Barrier(self.num_building)
            # Create shared memory for float64 data (enough for all processes' lists)
            shm = shared_memory.SharedMemory(create=True, size=shared_memory_array_size* np.float64().itemsize)

            # Run the EnergyPlus simulations in parallel for all buildings, monitored by the EnergyPlus API
            try:
                # Start tasks
                results_list = []
                with ProcessPoolExecutor(max_workers=self.num_building) as executor:
                    futures = [
                        executor.submit(
                            ep_simulation_instance.run_ep_simulation,
                            shm.name,
                            shared_memory_array_size,
                            shared_memory_lock,
                            synch_point_barrier,
                        )
                        for ep_simulation_instance in self._ep_simulation_instance_list
                    ]
                    # Wait for all processes to complete
                    for future in futures:
                        try:
                            results_list.extend(future.result())
                        except Exception as e:
                            print(f"Task generated an exception: {e}")
            finally:
                # Cleanup
                shm.close()
                shm.unlink()

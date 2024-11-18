"""
Class to manage the couped long-wave radiation (LWR) simulation with EnergyPlus among multiple buildings.
"""
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, Manager

from .ep_simulation_instance_with_shared_memory import EpSimulationInstance


class EpLwrSimulationManager:
    def __init__(self, simulation, coupling):
        # Simulation
        self._building_id_list = []
        self._ep_simulation_instance_list = []
        self._path_idf_list = []  # Ordered according to building_id_list


        # Input variables
        self._path_output_dir = None
        self._path_epw = None

        #


    #-----------------------------------------------------#
    #--------------------- Properties --------------------#
    #-----------------------------------------------------#

    @property
    def num_building(self):
        return len(self._building_id_list)


    def run_lwr_coupled_simulation(self):
        """

        :return:
        """

        #todo : identify the size of array_shape according to the number of outdoor surfaces of the buildings
        # need to be in the proper format
        array_shape= None

        # Run the simulation under a Manager context to share memory, locks, and barriers
        with Manager() as manager:

            # Initialize a lock to limit writing access to shared memory
            shared_memory_lock = manager.Lock()
            # Initialize a barrier to synchronize processes, when called with .wait() all processes will wait until all
            # processes have reached the barrier
            synch_point_barrier = manager.Barrier(self.num_building)

            # Create shared memory for float64 data (enough for all processes' lists)
            shm = shared_memory.SharedMemory(create=True, size=array_shape[0] * np.float64().itemsize)

            # Run the EnergyPlus simulations in parallel for all buildings, monitored by the EnergyPlus API
            try:
                # Start tasks
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            ep_simulation_instance.run_ep_simulation,
                            shm.name,
                            array_shape,
                            lock,
                            barrier,
                        )
                        for ep_simulation_instance in self._ep_simulation_instance_list
                    ]
                    # Wait for all processes to complete
                    for future in futures:
                        future.result()

            finally:
                # Cleanup
                shm.close()
                shm.unlink()
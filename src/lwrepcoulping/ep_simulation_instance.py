"""
Class to represent an instance of a simulation of the EP model, generating the idf additional strings,
managing the handlers etc.
"""

import os

from src.pyenergyplus.api import EnergyPlusAPI



class EPSimulationInstance:
    """

    """

    def __init__(self,path_idf:str, path_epw:str, path_output_dir:str, path_energyplus_dir:str):
        """

        """
        # initialize the EnergyPlus API and simulation state
        self.api = EnergyPlusAPI(running_as_python_plugin=True, path_to_ep_folder=path_energyplus_dir)
        self.state = self.api.state_manager.new_state()
        #
        self.path_idf = path_idf
        self.path_epw = path_epw
        self.path_output_dir = path_output_dir
        self.schedule_actuator_handle = None

        # Geometry
        self.outdoor_surface_list = []
        self.outdoor_surface_surrounding_surface_vf_dict = {}
        self.outdoor_surface_sky_vf_dict = {}
        self.outdoor_surface_ground_vf_dict = {}
        # todo: potentially need to add the emissivity or other related parameters

        # Handlers
        self.surface_temp_handler_dict = {}
        self.surface_surrounding_temperature_schedule_temperature_handler_dict = {}

        # LWR data
        self.f_epsilon_matrix = None
        self.f_matrix = None

        # Synchronization attributes

    def generate_idf_with_additional_strings(self):
        """

        :param additional_strings:
        :return:
        """

    def generate_additional_strings(self):
        """

        :return:
        """

    @staticmethod
    def read_other_building_surface_temperature():
        """

        :return:
        """

    def initialize_actuator_handler_callback_function(self):
        """

        :return:
        """

    def init_schedule_and_surface_temperature_handlers_call_back_function(self):
        """

        :return:
        """

    def coupled_simulation_callback_function(self):
        """
        Function to run at the end (or beginning) of each time step, to update the schedule values and surrounding surface temperatures.
        :return:
        """

        # Get current simulation time (in hours)

        # write down the surface temperatures in a file

        # wait for the other building to write down its surface temperatures

        # read the other building surface temperatures

        # update the surrounding surface temperature schedules with the proper "mean radiant temperature" values


    def run_ep_simulation(self):
        """

        :return:
        """


        # request the variables to access schedule and surface temperature values during the simulation
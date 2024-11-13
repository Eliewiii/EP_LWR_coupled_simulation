"""
Class to represent an instance of a simulation of the EP model, generating the idf additional strings,
managing the handlers etc.
"""

import os

from src.pyenergyplus.api import EnergyPlusAPI


class EPSimulationInstance:
    """

    """

    def __init__(self, path_idf: str, path_epw: str, path_output_dir: str, path_energyplus_dir: str,simulation_index: int):
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
        self.outdoor_surface_name_list = []
        self.outdoor_surface_surrounding_surface_vf_dict = {}
        self.outdoor_surface_sky_vf_dict = {}
        self.outdoor_surface_ground_vf_dict = {}
        # todo: potentially need to add the emissivity or other related parameters

        # Surrounding surface temperature schedule
        self.schedule_name_dict = {}

        # Handlers
        self.surface_temp_handler_dict = {}
        self.surrounding_surface_temperature_schedule_temperature_handler_dict = {}

        # LWR data
        self.f_epsilon_matrix = None
        self.f_matrix = None

        # Synchronization attributes
        self.simulation_index = simulation_index  # Index of the simulation, to synchronize the simulation and
        # set the order of surface temperature reading to match the LWR computation matrix

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

    def request_variables_before_running_simulation(self):
        """

        :return:

        A PRIORI DONE
        """
        for surface_name in self.outdoor_surface_name_list:
            self.api.exchange.get_variable_handle(self.state, "SURFACE OUTSIDE FACE TEMPERATURE",
                                                  surface_name)
            self.api.exchange.get_variable_handle(self.state, "Schedule Value",
                                                  self.schedule_name_dict[surface_name])

    def initialize_actuator_handler_callback_function(self):
        """

        :return:
        """
        for surface_name in self.outdoor_surface_name_list:
            schedule_actuator_handle = self.api.exchange.get_actuator_handle(self.state, "Schedule:Constant",
                                                                        "Schedule Value", self.schedule_name_dict[surface_name])
            if schedule_actuator_handle == -1:
                raise ValueError(f"Failed to create actuator for schedule {self.schedule_name_dict[surface_name]}")
            else:
                self.surrounding_surface_temperature_schedule_temperature_handler_dict[surface_name] = schedule_actuator_handle

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

        # Set the callback functions to run at the various moment of the simulation
        self.api.runtime.callback_after_new_environment_warmup_complete(self.state,
                                                                        self.initialize_actuator_handler_callback_function)
        self.api.runtime.callback_after_new_environment_warmup_complete(self.state,
                                                                        self.init_schedule_and_surface_temperature_handlers_call_back_function)
        self.api.runtime.callback_after_predictor_before_hvac_managers(self.state,
                                                                       self.coupled_simulation_callback_function)  # todo: might be change to the end of the timestep

        # Run the EnergyPlus simulation
        self.api.runtime.run_energyplus(self.state,
                                        ['-r',  # Run annual simulation
                                         '-w', self.path_epw,  # Weather file
                                         '-d', self.output_dir,  # Output directory
                                         self.idf_file]  # Input IDF file
                                        )

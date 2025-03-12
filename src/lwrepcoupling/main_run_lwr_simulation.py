"""

"""

import sys
import logging
import time

from lwrepcoupling import EpLwrSimulationManager

def main(path_pkl_file):
    """

    :param path_pkl_file:
    :return:
    """
    try:
        ep_lwr_sim_man = EpLwrSimulationManager.from_pkl(path_pkl_file)
    except Exception as e:
        raise Exception("Error loading the EpLwrSimulationManager object from the pkl file")
    # Check if enuogh buildings are in the simulation manager
    if ep_lwr_sim_man.num_building < 2:
        raise Exception("Not enough buildings in the simulation manager")
    #
    ep_lwr_sim_man.run_lwr_simulation()



if __name__ == "__main__":

    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level (DEBUG, INFO, WARNING, etc.)
        format="%(asctime)s - %(message)s"  # Customize your log message format
    )
    # Check if the configuration JSON file is provided
    if len(sys.argv) < 2:
        print("Please provide the configuration JSON file.")
    else:
        main(sys.argv[1])

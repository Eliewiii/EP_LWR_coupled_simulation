# Long Wave Radiation EnergyPlus Coupling
*Long Wave Radiation EnergyPlus Coupling* is a Python package that enables to couple the simulations between 
multiple buildings with EnergyPlus. It especially includes an accurate modeling of the long wave radiation
exchange between buildings, sharing the outdoor surfaces surface temperatures among all the buildings at each
timestep.



# Features


# Warning
This package includes inside the source code of pythonenergyplus, a Python package that enables to run the 
EnergyPlus API. This package is usually shipped with EnergyPlus, but cannot be called in a simple way from a 
random Python virtual environment. To bypass this issue the source code of pythonenergyplus was included 
directly in this package and slightly modified to point to the EnergyPlus executable. This is not a clean way
and will need to be addressed in the future.

# Pre-requisites
EnergyPlus(TM) needs to be installed on your computer. You can install it from the official website:
https://energyplus.net/downloads

EnergyPlusV23-2-0 was used for the development of this package, but it should work with later versions as well.

This package assumes that you already have EnergyPlus models files (.idf), weather files (.EPW) as well as the view factors among all the outdoor surfaces of the buildings are already computed.
A dedicated package, *Radiance Comp VF*, was developed for that purpose, but any other method can be used, 
as long as the view factors are provided to the *Long Wave Radiation EnergyPlus Coupling* package.



## License
This project is licensed under the MIT License, added the requirements of EnergyPlus(TM) for hhe use of pyenergyplus - see the LICENSE file for details.

## Contact
For any questions, feel free to reach out:

* Author: Elie MEIDONI
* Email: elie.medioniwiii@gmail.com

## Credits
A special thank to Ivan Girault for his help verifying the theoretical modeling anf long-wave radiation equation system. 

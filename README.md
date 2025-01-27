<!---
  AOgmaNeo
  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of AOgmaNeo is licensed to you under the terms described
  in the AOGMANEO_LICENSE.md file included in this distribution.
--->

# AOgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction 

Welcome to the [Ogma](https://ogmacorp.com) AOgmaNeo library, C++ library that contains an implementation of Sparse Predictive Hierarchies (SPH) aimed at desktop, embedded, and microcontroller devices.

For easier use on desktop, we also have [Python bindings](https://github.com/ogmacorp/PyAOgmaNeo).

This version of OgmaNeo (AOgmaNeo) is similar to [OgmaNeo2](https://github.com/ogmacorp/OgmaNeo2), but optimized in order to be able to run on an Arduino or similar microcontroller.
However, it has since become the default and preferred version of OgmaNeo for all tasks, as it works great on desktop as well.

For an introduction to how the algorithm works, see [the user guide](./AOgmaNeo_User_Guide.pdf).
For a more in-depth look, check out [the whitepaper](https://ogma.ai/sph-technology-description/).

## Installation

### CMake

Version 3.13+ of [CMake](https://cmake.org/) is required when building the library.

### OpenMP

This version of OgmaNeo uses [OpenMP](https://www.openmp.org/) for multiprocessing (on desktop). This will typically already be installed on your system.

### Building

The following commands can be used to build the AOgmaNeo library:

> git clone https://github.com/ogmacorp/AOgmaNeo.git  
> cd AOgmaNeo
> mkdir build  
> cd build  
> cmake ..  
> make  
> make install

The `cmake` command can be passed a `CMAKE_INSTALL_PREFIX` to determine where to install the library and header files.  

The `BUILD_SHARED_LIBS` boolean cmake option can be used to create dynamic/shared object library (default is to create a _static_ library). On Linux it's recommended to add `-DBUILD_SHARED_LIBS=ON` (especially if you plan to use the Python bindings in PyAOgmaNeo).

`make install` can be run to install the library. `make uninstall` can be used to uninstall the library.

On **Windows** systems it is recommended to use `cmake-gui` to define which generator to use and specify optional build parameters, such as `CMAKE_INSTALL_PREFIX`.

## Contributions

Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for information on making contributions to AOgmaNeo.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [AOGMANEO_LICENSE.md](./AOGMANEO_LICENSE.md) and [LICENSE.md](./LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

AOgmaNeo Copyright (c) 2020-2025 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.

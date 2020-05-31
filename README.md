<!---
  AOgmaNeo
  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of AOgmaNeo is licensed to you under the terms described
  in the AOGMANEO_LICENSE.md file included in this distribution.
--->

# AOgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction 

Welcome to the [Ogma](https://ogmacorp.com) AOgmaNeo library, C++ library that contains an implementation of Sparse Predictive Hierarchies for microcontroller devices.

This version of OgmaNeo (AOgmaNeo) is based on [OgmaNeo2](https://github.com/ogmacorp/OgmaNeo2), but stripped down in order to be able to run on an Arduino or similar microcontroller.
You can however also run it on a regular processor. That said, it is preferrable to use [regular OgmaNeo2](https://github.com/ogmacorp/OgmaNeo2) for regular processors to benefit from multithreading.

For an introduction to how the algorithm works, see [the presentation](https://github.com/ogmacorp/OgmaNeo2/blob/master/SPH_Presentation.pdf) (from the OgmaNeo2 repository).
For a more in-depth look, check out [the whitepaper](https://github.com/ogmacorp/OgmaNeo2/blob/master/OgmaNeo2_Whitepaper_DRAFT.pdf)  (from the OgmaNeo2 repository).

## Installation

Copy the source files to your Arduino sketch folder. Usage is similar to OgmaNeo2, except that you need to use the Array class instead of STL vectors.

## Memory Limits

AOgmaNeo will likely be limited by the amount of memory your microcontroller has. We used a fairly large microcontroller for testing, the Teensy 4.0. Speed is not so much an issue. Try to get at least 1MB of RAM.
Future versions of this software may provide options for alternative memory (e.g. MicroSD card or extended PSRAM chip, such as on the Teensy 4.1).

## Contributions

Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for information on making contributions to AOgmaNeo.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [AOGMANEO_LICENSE.md](./AOGMANEO_LICENSE.md) and [LICENSE.md](./LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

AOgmaNeo Copyright (c) 2020 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
# rtxAdvect - RTX-accelerated Tet-Mesh Particle Advection
==================================================================

# Building
==================================================================

## Dependencies

Optix 7, CUDA 10, and a recent NVIDIA driver.

## Checkout w/ Submodule

Mind this project uses a git submodule, so idealy clone with
`--recursive` flag:

    git clone --recursive git@gitlab.com:ingowald/rtxTetAdvect

If you did close without this flag, you can afterwards also do a

    git submodule init
	git submodule update
	
Building then via cmake and your favorite compiler toolchain.

## Building on Linux

For linux, I assume that `nvcc` is in the default path, and that there is environment variable names 'OptiX_INSTALL_DIR` that points to the optix SDK install directory

    cd rtxTetAdvect
	mkdir bin
	cd bin
	cmake ..
	make

If you have any missing dependencies ccmake will tell you... just fix and re-run cmake

## Building on Windows

For windows, install Visual Studio (2017+), CUDA (10.1+) and Optix 7 first. Then run `cmake-gui` and click configure with following
paramters:
`CUDA_HOST_COMPILER` =
`C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.26.28801/bin/Hostx64/x64/cl.exe`

`OptiX_INCLUDE` =
`C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0/include`

# License
==================================================================

Apache Licence 2.0 - pretty much do with it what you like; no warranties.

# Collaborators
==================================================================

UofU & NVidia
- Ingo Wald
- Will Usher
- Nate Morrical

LSU:
- Bin Wang

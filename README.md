# DN_Roof

# creat build folder
mkdir build
cd build
# create Visual Studio project files using cmake
cmake -G "Visual Studio 14 2015 Win64" ..
# build our application
cmake --build . --config Release
# once the build is complete, it will generate exe file in build\Release directory

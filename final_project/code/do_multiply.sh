git submodule init
git submodule update
cd build
cmake --build . --target clean
cmake . -DCMAKE_C_COMPILER=$(which gcc-8) -DCMAKE_CXX_COMPILER=$(which g++-8) -DWITH_CUDA=ON . 
make all
cd ..

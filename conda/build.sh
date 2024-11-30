mkdir -p ./build

cd ./build

cmake ../EigenIPC/ -DWITH_TESTS=OFF -DWITH_PYTHON=ON

make install
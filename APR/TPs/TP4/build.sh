#!/bin/bash
if [ -d linux ]; then
    echo "Build already exits"
else
    mkdir linux
fi
cd linux
cmake ..
cmake --build . --config Release --target exo1
cmake --build . --config Release --target exo2
cd ..

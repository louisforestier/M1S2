#!/bin/bash
TARGETS="PremierPas Communication NaiveBroadcast"
if [ -d linux ]; then
    echo '\e[1;32;41mFolder "linux" already exits\e[0m'
else
    mkdir linux
    mkdir linux/o1
    mkdir linux/o2
    mkdir linux/o3
fi
cd linux

echo '\e[1;32;41mBuild "PremierPas"\e[0m'
mpicxx -c ../src/1-PremierPas/main.cc -W -Wall -o ./o1/main.o
mpicxx -o PremierPas ./o1/main.o -W -Wall

echo '\e[1;32;41mBuild "Communication"\e[0m'
mpicxx -c ../students/2-Communication/main.cc -W -Wall -o ./o2/main.o
mpicxx -o Communication ./o2/main.o -W -Wall

echo '\e[1;32;41mBuild "NaiveBroadcast"\e[0m'
mpicxx -c ../src/3-1-broadcast/main.cc -I../src -I../students -W -Wall -o ./o3/main.naive.o
mpicxx -c ../students/3-1-broadcast/Broadcast.cc  -I../src -I../students -W -Wall -o ./o3/Broadcast.naive.o
mpicxx -c ../src/utils/chronoCPU.cc  -I../src -I../students -W -Wall -o ./o3/chronoCPU.o
mpicxx -o NaiveBroadcast ./o3/main.naive.o ./o3/Broadcast.naive.o ./o3/chronoCPU.o

echo '\e[1;32;41mBuild "PipeBroadcast"\e[0m'
mpicxx -c ../src/3-2-broadcast/main.cc  -I../src -I../students -W -Wall -o ./o3/main.pipe.o
mpicxx -c ../students/3-2-broadcast/Broadcast.cc  -I../src -I../students -W -Wall -o ./o3/Broadcast.pipe.o
mpicxx -o PipeBroadcast ./o3/main.pipe.o ./o3/Broadcast.pipe.o ./o3/chronoCPU.o



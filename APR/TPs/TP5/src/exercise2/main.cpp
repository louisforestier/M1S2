#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise2/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    // run exercise 3
    ExerciseImpl("Exercise 2 : tri base \"sequentiel\"")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

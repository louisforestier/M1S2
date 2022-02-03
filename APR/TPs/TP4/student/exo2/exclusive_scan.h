#pragma once
#include <thread>
#include <vector>
#include <OPP.h>
#include <algorithm>

// inclusive scan

namespace OPP {
    
    template<   typename InputIteratorType, 
                typename OutputIteratorType, 
                typename BinaryFunction,
                typename T >
    inline
    void exclusive_scan(
        const InputIteratorType&& aBegin, // input begin
        const InputIteratorType&& aEnd,   // input end (excluded)
        const OutputIteratorType&& oBegin, // output begin
        const BinaryFunction&& functor, // should be associative
        const T Tinit = T(0)
    ) {
        // TODO
    }
};
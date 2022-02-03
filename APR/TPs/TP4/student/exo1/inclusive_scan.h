#pragma once
#include <thread>
#include <vector>
#include <OPP.h>
#include <algorithm>
#include <iterator>

// inclusive scan

namespace OPP {
    
    template<   typename InputIteratorType, 
                typename OutputIteratorType, 
                typename BinaryFunction >
    inline
    void inclusive_scan(
        const InputIteratorType&& aBegin, // input begin
        const InputIteratorType&& aEnd,   // input end (excluded)
        const OutputIteratorType&& oBegin, // output begin
        const BinaryFunction&& functor // should be associative
    ) {
        // TODO ...

    }
};
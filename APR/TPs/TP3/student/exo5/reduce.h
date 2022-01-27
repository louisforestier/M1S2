#pragma once
#include <OPP.h>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <ranges>

// gather is a permutation of data. The source index is given thanks to an iterator. 

namespace OPP {
    
    template<   typename InputIteratorType, 
                typename T, 
                typename MapFunction>
        inline
    T reduce(
        const InputIteratorType&& aBegin, 
        const InputIteratorType&& aEnd,
        const T&& init,
        const MapFunction&& functor // unary functor
    ) {
        // TODO: 
        return T(0);
    }
};
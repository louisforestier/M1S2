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
                typename MapIteratorType, 
                typename OutputIteratorType>
        inline
    void gather(
        const InputIteratorType&& aBegin, // left operand
        const InputIteratorType&& aEnd,
        const MapIteratorType&& map, // source index
        OutputIteratorType&& oBegin // destination
    ) {
        // TODO: a map using OPP::nbThreads threads
        // that does something like:
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //   oBegin[iter-aBegin] = iter[map[iter-aBegin]];
        
        // chunk size
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + OPP::nbThreads-1) / OPP::nbThreads;
        // launch the threads
        std::vector<std::thread> threads(OPP::nbThreads);

        // ...
    }
};
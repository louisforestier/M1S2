#pragma once

#include <OPP.h>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <ranges>

namespace OPP
{
    template<   typename InputIteratorType, 
                typename OutputIteratorType, 
                typename MapFunction> 
        inline
    void transform(
        const InputIteratorType&& aBegin, // left operand
        const InputIteratorType&& aEnd,
        OutputIteratorType&& oBegin, // destination
        const MapFunction&& functor // unary functor
    ) {
        // TODO: a map using OPP::nbThreads threads
        //  that does something like the following
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //    oBegin[iter-aBegin] = functor(*iter);
        // chunk size
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + OPP::nbThreads-1) / OPP::nbThreads;
        // launch the threads/tasks
        // TODO
        OPP::ThreadPool& pool = OPP::getDefaultThreadPool();
        size_t nb_threads = pool.getRealThreadNumber();
        for (auto i = 0; i < nb_threads; ++i)
        {
            pool.push_task(
                [i](){
                    for (auto n = i; i < fullSize; n+=nb_threads)
                    {
                        
                    }
                    
                }
            );
        }
    }

 
    // second version: two input iterators!
    template<   typename InputIteratorType, 
                typename OutputIteratorType, 
                typename MapFunction>
        inline
    void transform(
        const InputIteratorType&& aBegin, // left operand
        const InputIteratorType&& aEnd,
        const InputIteratorType&& bBegin, // right operand
        OutputIteratorType&& oBegin, // destination
        const MapFunction&& functor // binary functor
    ) {
        // TODO: a map using OPP::nbThreads threads
        // that does something like:
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //   oBegin[iter-aBegin] = functor(*iter, bBegin[iter-aBegin]);
        
        // chunk size
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + OPP::nbThreads-1) / OPP::nbThreads;
        // launch the threads/Tasks
        // TODO
    }
};
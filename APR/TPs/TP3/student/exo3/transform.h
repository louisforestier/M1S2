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
        OPP::Semaphore<uint32_t> semaphore(0);
        int nb_tasks = 4 * OPP::nbThreads;
        for (int i = 0; i < nb_tasks; ++i)
        {
            pool.push_task(
                [i,nb_tasks,&semaphore,&aBegin,&aEnd,&oBegin,&functor](){
                    for (auto iter = aBegin+i; iter < aEnd; iter+=nb_tasks)
                    {
                        oBegin[iter-aBegin] = functor(*iter);
                    }
                    semaphore.release();
                }
            );
        }
        semaphore.acquire(nb_tasks);
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
        OPP::ThreadPool& poule_de_freud = OPP::getDefaultThreadPool();
        OPP::Semaphore<uint32_t> semaphore(0);
        int nb_tasks = 4 * OPP::nbThreads;
        for (int i = 0; i < nb_tasks; ++i)
        {
            pool.push_task(
                [i,nb_tasks,&semaphore,&aBegin,&aEnd,&bBegin,&oBegin,&functor](){
                    for (auto iter = aBegin+i; iter < aEnd; iter+=nb_tasks)
                    {
                        oBegin[iter-aBegin] = functor(*iter,bBegin[iter-aBegin]);
                    }
                    semaphore.release();
                }
            );
        }
        semaphore.acquire(nb_tasks);
    }
};
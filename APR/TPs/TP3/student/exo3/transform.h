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
        // a map using OPP::nbThreads threads
        //  that does something like the following
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //    oBegin[iter-aBegin] = functor(*iter);
        // chunk size
        int nb_tasks = 4 * OPP::nbThreads;
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + nb_tasks-1) / nb_tasks;
        // launch the threads/tasks
        OPP::ThreadPool& pool = OPP::getDefaultThreadPool();
        OPP::Semaphore<uint32_t> semaphore(0);
        for (int i = 0; i < nb_tasks; ++i)
        {
            //stratégie modulo
            pool.push_task(
                [i,nb_tasks,&semaphore,&aBegin,&aEnd,&oBegin,&functor](){
                    for (auto iter = aBegin+i; iter < aEnd; iter+=nb_tasks)
                    {
                        oBegin[iter-aBegin] = functor(*iter);
                    }
                    semaphore.release();
                }
            );

            //stratégie par bloc
            /* auto end = std::min((i+1)*chunkSize, fullSize);
            pool.push_task(
                [i,chunkSize,end,&semaphore,&aBegin,&oBegin,&functor](){
                    for (auto iter = aBegin+i*chunkSize; iter < aBegin+end; iter++)
                    {
                        oBegin[iter-aBegin] = functor(*iter);
                    }
                    semaphore.release();
                } 
            ); */
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
        // a map using OPP::nbThreads threads
        // that does something like:
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //   oBegin[iter-aBegin] = functor(*iter, bBegin[iter-aBegin]);
        
        // chunk size
        int nb_tasks = 4 * OPP::nbThreads;
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + nb_tasks-1) / nb_tasks;
        // launch the threads/Tasks
        OPP::ThreadPool& pool = OPP::getDefaultThreadPool();
        OPP::Semaphore<uint32_t> semaphore(0);
        for (int i = 0; i < nb_tasks; ++i)
        {
            //stratégie modulo
            pool.push_task(
                [i,nb_tasks,&semaphore,&aBegin,&aEnd,&bBegin,&oBegin,&functor](){
                    for (auto iter = aBegin+i; iter < aEnd; iter+=nb_tasks)
                    {
                        oBegin[iter-aBegin] = functor(*iter,bBegin[iter-aBegin]);
                    }
                    semaphore.release();
                }
            );

            //stratégie par bloc
            /* auto end = std::min((i+1)*chunkSize, fullSize);
            pool.push_task(
                [i,chunkSize,end,&semaphore,&aBegin,&bBegin,&oBegin,&functor](){
                    for (auto iter = aBegin+i*chunkSize; iter < aBegin+end; iter++)
                    {
                        oBegin[iter-aBegin] = functor(*iter,bBegin[iter-aBegin]);
                    }
                    semaphore.release();
                } 
            ); */
        }
        semaphore.acquire(nb_tasks);
    }
};
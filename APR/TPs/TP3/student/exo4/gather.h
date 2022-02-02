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
        // a map using OPP::nbThreads threads
        // that does something like:
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //   oBegin[iter-aBegin] = iter[map[iter-aBegin]];
        
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
                [i,nb_tasks,&semaphore,&aBegin,&aEnd,&oBegin,&map](){
                    for (auto iter = aBegin+i; iter < aEnd; iter+=nb_tasks)
                    {
                        oBegin[iter-aBegin] = aBegin[map[iter-aBegin]];
                    }
                    semaphore.release();
                }
            );

            //stratégie par bloc
            /* auto end = std::min((i+1)*chunkSize, fullSize);
            pool.push_task(
                [i,chunkSize,end,&semaphore,&aBegin,&oBegin,&map](){
                    for (auto iter = aBegin+i*chunkSize; iter < aBegin+end; iter++)
                    {
                        oBegin[iter-aBegin] = aBegin[map[iter-aBegin]];
                    }
                    semaphore.release();
                } 
            ); */

        }
        semaphore.acquire(nb_tasks); 
    }
};
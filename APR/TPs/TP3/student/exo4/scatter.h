#pragma once
#include <OPP.h>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <ranges>

// scatter is a permutation of data. The destination index is given thanks to an iterator. 

namespace OPP {
        
    template<   typename InputIteratorType, 
                typename MapIteratorType, 
                typename OutputIteratorType>
        inline
    void scatter(
        const InputIteratorType&& aBegin, // left operand
        const InputIteratorType&& aEnd,
        const MapIteratorType&& map, // source index
        OutputIteratorType&& oBegin // destination
    ) {
        // TODO: a map using OPP::nbThreads threads that does something like:
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //   oBegin[map[iter-aBegin]] = iter[iter-aBegin];
        
        // chunk size
        // auto fullSize = aEnd - aBegin;
        // auto chunkSize = (fullSize + OPP::nbThreads-1) / OPP::nbThreads;
        // launch the threads/tasks
        // TODO
        OPP::ThreadPool& poule_de_freud = OPP::getDefaultThreadPool();
        OPP::Semaphore<uint32_t> semaphore(0);
        int nb_tasks = 4 * OPP::nbThreads;
        for (int i = 0; i < nb_tasks; ++i)
        {
            poule_de_freud.push_task(
                [i,nb_tasks,&semaphore,&aBegin,&aEnd,&oBegin,&map](){
                    for (auto iter = aBegin+i; iter < aEnd; iter+=nb_tasks)
                    {
                        oBegin[map[iter-aBegin]] = aBegin[iter-aBegin];
                    }
                    semaphore.release();
                }
            );
        }
        semaphore.acquire(nb_tasks);
     }
};
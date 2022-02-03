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
        const MapFunction&& functor // binary functor
    ) {

        int nb_tasks = 4 * OPP::nbThreads;
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + nb_tasks-1) / nb_tasks;
        OPP::ThreadPool& pool = OPP::getDefaultThreadPool();
        OPP::Semaphore<uint32_t> semaphore(0);
        std::vector<T> results(nb_tasks);
        for (int i = 0; i < nb_tasks; ++i)
        {
            //stratégie modulo
            /* pool.push_task(
                [i,nb_tasks,&init,&semaphore,&aBegin,&aEnd,&functor,&results](){
                    results[i] = init;
                    for (auto iter = aBegin+i; iter < aEnd; iter+=nb_tasks)
                    {
                        results[i]=functor(results[i],aBegin[iter-aBegin]);
                    }
                    semaphore.release();
                }
            ); */

            //stratégie par bloc
            auto end = std::min((i+1)*chunkSize, fullSize);
            pool.push_task(
                [i,chunkSize,end,&semaphore,&aBegin,&functor,&results](){
                    for (auto iter = aBegin+i*chunkSize; iter < aBegin+end; iter++)
                    {
                        results[i]=functor(results[i],aBegin[iter-aBegin]);
                    }
                    semaphore.release();
                } 
            );

        }
        semaphore.acquire(nb_tasks);
        T res = results[0];
        for (int i = 1; i < nb_tasks; i++)
        {
            res = functor(res, results[i]);
        }
        
        return res;
    }
};
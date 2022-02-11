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
        int nb_tasks = 4 * OPP::nbThreads;
        auto fullSize = aEnd - aBegin;
        auto blockSize = (fullSize + nb_tasks - 1) / nb_tasks;

        OPP::ThreadPool &pool = OPP::getDefaultThreadPool();
        OPP::Semaphore<uint32_t> semaphore(0);

        for (int i = 0; i < nb_tasks; ++i)
        {
            pool.push_task(
                [i,Tinit,blockSize, fullSize ,&semaphore,&aBegin,&oBegin,&functor](){
                    auto blockStart = blockSize * i;
                    if (blockStart < fullSize)
                    {
                        oBegin[blockStart] = Tinit;
                        for (int j = 1; j < blockSize; j++)
                        {                            
                                oBegin[blockStart+j] = functor(oBegin[blockStart+j-1],aBegin[blockStart+j-1]);
                            
                        }
                    }
                    semaphore.release();
                }
            ); 
        }
        semaphore.acquire(nb_tasks);
        std::vector<T>aux(nb_tasks-1);
        aux[0] = functor(oBegin[blockSize-1],aBegin[blockSize-1]);
        for (int i = 1; i < nb_tasks && blockSize * i < fullSize; i++)
        {
            aux[i] = functor(functor(aux[i-1],oBegin[blockSize*(i+1)-1]),aBegin[blockSize*(i+1)-1]);
        }

        for (int i = 0; i < nb_tasks-1; i++)
        {
            pool.push_task(
                [i,blockSize,fullSize,&semaphore,&oBegin,&aux,&functor](){

                    auto blockStart = blockSize * (i+1);
                    if (blockStart < fullSize)
                    {
                        for (int j = 0; j < blockSize ; j++)
                        {
                            if (blockStart+j < fullSize)
                            {
                                oBegin[blockStart+j] = functor(aux[i], oBegin[blockStart+j]);
                            }
                        }
                    }
                    semaphore.release();
                }
            );
        }
        semaphore.acquire(nb_tasks-1);
    }
};
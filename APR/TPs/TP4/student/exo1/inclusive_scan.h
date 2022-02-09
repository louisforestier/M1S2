#pragma once
#include <thread>
#include <vector>
#include <OPP.h>
#include <algorithm>
#include <iterator>

// inclusive scan

namespace OPP
{

    template <typename InputIteratorType,
              typename OutputIteratorType,
              typename BinaryFunction>
    inline void inclusive_scan(
        const InputIteratorType &&aBegin,  // input begin
        const InputIteratorType &&aEnd,    // input end (excluded)
        const OutputIteratorType &&oBegin, // output begin
        const BinaryFunction &&functor     // should be associative
    )
    {
        int nb_tasks = 4 * OPP::nbThreads;
        auto fullSize = aEnd - aBegin;
        auto blockSize = (fullSize + nb_tasks - 1) / nb_tasks;

        OPP::ThreadPool &pool = OPP::getDefaultThreadPool();
        OPP::Semaphore<uint32_t> semaphore(0);

        for (int i = 0; i < nb_tasks; ++i)
        {
            pool.push_task(
                [i,blockSize, fullSize ,&semaphore,&aBegin,&oBegin,&functor](){
                    auto blockStart = blockSize * i;
                    if (blockStart < fullSize)
                    {
                        oBegin[blockStart] = aBegin[blockStart];
                        for (int j = 1; j < blockSize; j++)
                        {
                            if (blockStart+j < fullSize)
                            {
                                oBegin[blockStart+j] = functor(oBegin[blockStart+j-1],aBegin[blockStart+j]);
                            }
                        }
                    }
                    semaphore.release();
                }
            ); 
        }
        semaphore.acquire(nb_tasks);

        std::vector<typename OutputIteratorType::value_type>aux(nb_tasks-1);
        aux[0] = oBegin[blockSize-1];
        for (int i = 1; i < nb_tasks && blockSize * i < fullSize; i++)
        {
            aux[i] = functor(aux[i-1],oBegin[blockSize*(i+1)-1]);
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
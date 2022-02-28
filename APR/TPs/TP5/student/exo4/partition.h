#pragma once
#include <thread>
#include <vector>
#include <OPP.h>
#include <algorithm>

// partition

namespace OPP 
{

    template<   typename InputIteratorType, 
                typename PredicateIteratorType, 
                typename OutputIteratorType >
    inline
    void partition(
        const InputIteratorType&& aBegin, // input begin
        const InputIteratorType&& aEnd,   // input end (excluded)
        const PredicateIteratorType&& predBegin,   // predicate begin, should be iterator on int ...
        const OutputIteratorType&& oBegin // output begin
    ) {
        // chunk size
        const auto fullSize = static_cast<decltype(nbThreads)>(aEnd - aBegin);
        const auto realNbThreads = std::min(fullSize, nbThreads);
        const auto chunkSize = (fullSize + realNbThreads-1) / realNbThreads;
        
        using T = typename InputIteratorType::value_type;

        Barrier barrier(realNbThreads);
        
        std::vector<typename OutputIteratorType::value_type> sumsTrue(fullSize);
        std::vector<typename OutputIteratorType::value_type> sumsFalse(fullSize);
        
        std::vector<typename OutputIteratorType::value_type> partialSumsTrue(realNbThreads);
        std::vector<typename OutputIteratorType::value_type> partialSumsFalse(realNbThreads);
        
        
        auto fun_thread = [&] (
            const size_t begin, 
            const size_t end,
            const unsigned thread_num
        ) -> void 
        {
            // TODO : travailler entre begin et end (des entiers) dans les tableaux
            // NB : ici tout ce qui est visible dans la fonction partition est captur� (donc utilisable)
            if (begin < fullSize)
            {
                sumsFalse[begin] = typename OutputIteratorType::value_type(0);
                sumsTrue[end-1] = predBegin[end-1];
                for (int i = 1; i < chunkSize; i++)
                {
                    if (begin + i < fullSize)
                    {
                        sumsFalse[begin+i] = sumsFalse[begin+i-1] + (!predBegin[begin+i-1]);
                        sumsTrue[end-1-i] = sumsTrue[end-i] + predBegin[end-1-i];
                    }
                    
                }
                
            }
            std::cout<<"thread"<<thread_num<<" avant b1"<<std::endl;
            barrier.arrive_and_wait();
            std::cout<<"thread"<<thread_num<<" apres b1"<<std::endl;
            if (thread_num == 0)
            {
                partialSumsFalse[0] = sumsFalse[chunkSize-1]+(!predBegin[chunkSize-1]);
                partialSumsTrue[end-1] = sumsTrue[end - chunkSize];
                for (int i = 1; i < realNbThreads && chunkSize * i < fullSize ; i++)
                {
                    partialSumsFalse[i] = partialSumsFalse[i-1] + sumsFalse[chunkSize*(i+1)-1] + (!predBegin[chunkSize*(i+1)-1]);
                    partialSumsTrue[end-1-i] = partialSumsTrue[end-1-i] + sumsTrue[end-1-chunkSize*(i+1)] ;
                }
            }
            std::cout<<"thread"<<thread_num<<" avant b2"<<std::endl;
            barrier.arrive_and_wait();
            std::cout<<"thread"<<thread_num<<" apres b2"<<std::endl;
            if (begin < fullSize)
            {
                for (int i = 0; i < chunkSize; i++)
                {
                    if (begin + i < fullSize)
                    {
                        sumsFalse[begin+i] = partialSumsFalse[i] + sumsFalse[begin+i];
                    }
                    
                }
                
            }
            barrier.arrive_and_wait();

        };
        std::cout<<"nb threads="<<realNbThreads<<std::endl;
        // launch the threads
        std::vector<std::thread> threads(realNbThreads);
        for(auto i=0u; i<realNbThreads; i++) {
            threads[i] = 
                std::thread(
                    fun_thread,
                    i*chunkSize, 
                    std::min((i+1)*chunkSize, fullSize), 
                    i
                );
        };
        std::cout<<"avant join"<<std::endl;
        for(auto& th : threads)
            th.join();
        std::cout<<"apres join"<<std::endl;
    }
};
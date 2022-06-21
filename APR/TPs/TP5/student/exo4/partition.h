#pragma once
#include <thread>
#include <vector>
#include <OPP.h>
#include <algorithm>

// partition

namespace OPP
{

    template <typename InputIteratorType,
              typename PredicateIteratorType,
              typename OutputIteratorType>
    inline void partition(
        const InputIteratorType &&aBegin,        // input begin
        const InputIteratorType &&aEnd,          // input end (excluded)
        const PredicateIteratorType &&predBegin, // predicate begin, should be iterator on int ...
        const OutputIteratorType &&oBegin        // output begin
    )
    {
        // chunk size
        const auto fullSize = static_cast<decltype(nbThreads)>(aEnd - aBegin);
        const auto realNbThreads = std::min(fullSize, nbThreads);
        const auto chunkSize = (fullSize + realNbThreads - 1) / realNbThreads;

        using T = typename InputIteratorType::value_type;

        Barrier barrier(realNbThreads);
        Barrier barrier2(realNbThreads);
        Barrier barrier3(realNbThreads);

        std::vector<typename OutputIteratorType::value_type> sumsTrue(fullSize);
        std::vector<typename OutputIteratorType::value_type> sumsFalse(fullSize);

        std::vector<typename OutputIteratorType::value_type> partialSumsTrue(realNbThreads);
        std::vector<typename OutputIteratorType::value_type> partialSumsFalse(realNbThreads);

        auto fun_thread = [&](
            const size_t begin,
            const size_t end,
            const unsigned thread_num) -> void
        {
            // TODO : travailler entre begin et end (des entiers) dans les tableaux
            // NB : ici tout ce qui est visible dans la fonction partition est captur� (donc utilisable)
            if (begin < fullSize)
            {
                sumsTrue[begin] = typename OutputIteratorType::value_type(0);
                sumsFalse[end - 1] = !(predBegin[end - 1]);
                for (unsigned int i = 1; i < chunkSize; i++)
                {
                    if (begin + i < fullSize)
                    {
                        sumsTrue[begin + i] = sumsTrue[begin + i - 1] + predBegin[begin + i - 1];
                        sumsFalse[end - i - 1] = sumsFalse[end - i] + !predBegin[end - 1 - i];
                    }
                }
            }
            barrier.arrive_and_wait();
            if (thread_num == realNbThreads - 1)
            {
                partialSumsTrue[0] = sumsTrue[chunkSize - 1] + predBegin[chunkSize - 1];
                partialSumsFalse[0] = sumsFalse[begin];
                for (unsigned int i = 1; i < realNbThreads && chunkSize * (i + 1) < fullSize; i++)
                {
                    partialSumsTrue[i] = partialSumsTrue[i - 1] + sumsTrue[chunkSize * (i + 1) - 1] + predBegin[chunkSize * (i + 1) - 1];
                    partialSumsFalse[i] = partialSumsFalse[i - 1] + sumsFalse[begin - chunkSize * (i + 1)];
                }
            }
            barrier2.arrive_and_wait();
            if (begin + chunkSize < fullSize)
            {
                for (unsigned int i = 0; i < chunkSize; i++)
                {
                    if (begin + chunkSize + i < fullSize)
                    {
                        sumsTrue[begin + chunkSize + i] = partialSumsTrue[thread_num] + sumsTrue[begin + chunkSize + i];
                    }
                    if (end - i - 1 >= 0)
                    {
                        sumsFalse[end - i - 1] = partialSumsFalse[realNbThreads - thread_num - 2] + sumsFalse[end - i - 1];
                    }
                }
            }
            barrier3.arrive_and_wait();
            if (begin < fullSize)
            {
                for (size_t i = begin; i < end; i++)
                {
                    if (predBegin[i])
                    {
                        oBegin[sumsTrue[i]] = aBegin[i];
                    }
                    else
                    {
                        oBegin[fullSize - sumsFalse[i]] = aBegin[i];
                    }
                }
            }
        };
        // launch the threads
        std::vector<std::thread> threads(realNbThreads);
        for (auto i = 0u; i < realNbThreads; i++)
        {
            threads[i] =
                std::thread(
                    fun_thread,
                    i * chunkSize,
                    std::min((i + 1) * chunkSize, fullSize),
                    i);
        };
        for (auto &th : threads)
            th.join();
    }
};
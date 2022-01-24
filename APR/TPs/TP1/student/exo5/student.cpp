//FORESTIER Louis
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <iostream>
#include "student.h"

namespace
{
	class MoniteurIntervalle
	{
	private:
		uint32_t current;
		const uint32_t max;
		std::mutex *mutex;

	public:
		MoniteurIntervalle(const uint32_t min, const uint32_t max)
			: current(min), max(max), mutex(new std::mutex())
		{
		}
		~MoniteurIntervalle()
		{
			delete mutex;
		}

		uint32_t getCurrent()
		{
			uint32_t curr;
			mutex->lock();
			curr = current;
			current++;
			mutex->unlock();
			return curr;
		}
		
		uint32_t getMax() const
		{
			return max;
		}
	};

	class MoniteurPremier
	{
	private:
		std::vector<std::pair<uint32_t, uint32_t>> pairs;
		std::mutex *mutex;

	public:
		MoniteurPremier()
			: pairs(), mutex(new std::mutex())
		{
		}
		~MoniteurPremier()
		{
			delete mutex;
		}

		void addPair(uint32_t a, uint32_t b)
		{
			mutex->lock();
			pairs.emplace_back(std::pair<uint32_t, uint32_t>(a, b));
			mutex->unlock();
		}

		std::vector<std::pair<uint32_t, uint32_t>> getPairs()
		{
			std::vector<std::pair<uint32_t, uint32_t>> result;
			mutex->lock();
			result = pairs;
			mutex->unlock();
			return result;
		}
	};

	bool are_2_pairs_sorted(const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b)
	{
		return std::get<0>(a) < std::get<0>(b);
	}

	bool is_prime(const uint32_t n)
	{
		// check division from 2 to n (not efficient at all!)
		for (uint32_t d = 2; d < n; ++d)
			if ((n % d) == 0) // d is a divisor, n is not prime
				return false;
		// we have not found any divisor: n is prime
		return true;
	}

	void calculate(MoniteurIntervalle* moniteurI, MoniteurPremier* moniteurP)
	{
		uint32_t curr = moniteurI->getCurrent();
		uint32_t max = moniteurI->getMax();
		while (curr < max-1)
		{
			if (is_prime(curr))
			{
				if (is_prime(curr + 2))
				{
					moniteurP->addPair(curr, curr + 2);
				}
			}
			curr = moniteurI->getCurrent();			
		}		
	}

	std::vector<std::pair<uint32_t, uint32_t>>
	computePairs(const uint32_t min, const uint32_t max, const unsigned nb_threads)
	{
		MoniteurIntervalle moniteurI(min, max);
		MoniteurPremier moniteurP;
		std::cout << "starting " << nb_threads << " threads ..." << std::endl;
		std::vector<std::thread> threads;
		for (unsigned int i = 0; i < nb_threads; ++i)
		{
			threads.emplace_back(std::thread(calculate, &moniteurI, &moniteurP));
		}
		for (unsigned int i = 0; i < nb_threads; ++i)
			threads[i].join();
		std::cout << "threads have completed." << std::endl;
		std::vector<std::pair<uint32_t, uint32_t>> result = moniteurP.getPairs();
		std::sort(result.begin(), result.end(), are_2_pairs_sorted);
		return result;
	}

}

bool StudentWork5::isImplemented() const
{
	return true;
}

std::vector<std::pair<uint32_t, uint32_t>>
StudentWork5::run(const uint32_t min, const uint32_t max, const uint32_t nb_threads)
{
	std::vector<std::pair<uint32_t, uint32_t>> result;
	result = computePairs(min, max, nb_threads);
	return result;
}

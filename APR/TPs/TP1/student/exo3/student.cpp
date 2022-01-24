//FORESTIER Louis
#include <thread> // C++ 11
#include <mutex> // C++ 11
#include <iostream>
#include <cmath>
#include <vector>
#include <exo3/student.h>


namespace {

	double results=0.0;
	std::mutex mutex;
	const unsigned int limit = 1 << 28; // 2^28 == 256 millions

	double compute_pi(const unsigned nb_threads)
	{
		std::cout << "starting " << nb_threads << " threads ..." << std::endl;
		std::vector<std::thread> threads;
		for (unsigned int i = 0; i < nb_threads; ++i){
			threads.emplace_back(std::thread(
				[i,nb_threads]()
				{
					double acc = 0.0;
					for (unsigned int n = i; n < limit; n += nb_threads)
					{
						acc += pow(-1.0, n) / (2.0 * n + 1.0);
					}
					mutex.lock();
					results += acc;
					mutex.unlock();
				} 
			));
		};
		// synchronize threads:
		for (unsigned int i = 0; i < nb_threads; ++i)
			threads[i].join();
		std::cout << "threads have completed." << std::endl;
		double pi = results;
		pi *= 4.0;
		std::cout.precision(12);
		std::cout << "our evaluation is: " << std::fixed << pi << std::endl;
		return pi;
	}
}

bool StudentWork3::isImplemented() const {
	return true;
}

/// nb_threads is between 1 to 64 ...
double StudentWork3::run(const unsigned nb_threads) 
{
	return compute_pi(nb_threads);
}

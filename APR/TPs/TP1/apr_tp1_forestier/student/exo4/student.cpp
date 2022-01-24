//FORESTIER Louis
#include <thread> // C++ 11
#include <mutex>  // C++ 11
#include <iostream>
#include <cmath>
#include <vector>
#include <exo4/student.h>

namespace
{

	class MoniteurPi
	{
	private:
		double result;
		std::mutex *mutex;

	public:

		MoniteurPi()
		: result(0.0),mutex(new std::mutex())
		{}
		
		~MoniteurPi(){
			delete mutex;
		}

		double getResult() const {
			double resultat;
			mutex->lock();
			resultat = result;
			mutex->unlock();
			return resultat;
		}

		void add(const double acc){
			mutex->lock();
			this->result += acc;
			mutex->unlock();
		}

	};


	const unsigned int limit = 1 << 28; // 2^28 == 256 millions
	MoniteurPi moniteur;

	double compute_pi(const unsigned nb_threads)
	{
		std::cout << "starting " << nb_threads << " threads ..." << std::endl;
		std::vector<std::thread> threads;
		for (unsigned int i = 0; i < nb_threads; ++i)
		{
			threads.emplace_back(std::thread(
				[i, nb_threads]()
				{
					double acc = 0.0;
					for (unsigned int n = i; n < limit; n += nb_threads)
					{
						acc += pow(-1.0, n) / (2.0 * n + 1.0);
					}
					moniteur.add(acc);
				}));
		};
		// synchronize threads:
		for (unsigned int i = 0; i < nb_threads; ++i)
			threads[i].join();
		std::cout << "threads have completed." << std::endl;
		double pi = moniteur.getResult();
		pi *= 4.0;
		std::cout.precision(12);
		std::cout << "our evaluation is: " << std::fixed << pi << std::endl;
		return pi;
	}

}

bool StudentWork4::isImplemented() const
{
	return true;
}

/// nb_threads is between 1 to 64 ...
double StudentWork4::run(const unsigned nb_threads)
{
	return compute_pi(nb_threads);
}

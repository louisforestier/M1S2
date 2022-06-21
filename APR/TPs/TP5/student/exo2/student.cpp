#include <iostream>
#include <functional>
#include <exo2/student.h>
#include <algorithm>

namespace 
{
	template<typename T>
	inline unsigned extract(T value, unsigned bit)
	{
		if (bit == 0)
		{
			return 1-(value & 0x1);
		}
		return 1 - ((value >> bit) & 0x1);
	}

	//PRECONDITION
	//predicate doit etre de la meme taille que input
	template<typename T>
	void fill_predicate(std::vector<T> & input, std::vector<unsigned>& predicate, unsigned numeroBit)
	{
		for (unsigned i = 0; i < input.size(); i++)
		{
			predicate[i] = extract(input[i], numeroBit);
		}
		
	}
	template<typename T>
	void partition(std::vector<T>& input, std::vector<T>& output, std::vector<unsigned>& predicate)
	{
		unsigned j = 0;
		for (unsigned i = 0; i < input.size(); i++)
		{
			if (predicate[i])
			{
				output[j++] = input[i];
			}
		}
		for (unsigned i = 0; i < input.size(); i++)
		{
			if (!(predicate[i]))
			{
				output[j++] = input[i];
			}
		}
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_radixSort_sequential(
	std::vector<unsigned>& input,
	std::vector<unsigned>& output
) {
	// utiliser l'algorithme vu en court/TD
	// pour chaque bit, en partant du poids faible
	//   calculer predicat = i�me bit (c'est un MAP, s�quentiel ici)
	//   partitionner (s�quentiellement)
	// ... et c'est tout !
	// Attention quand m�me : le partitionnement n�cessite un tableau auxiliaire !!!
	// Le plus simple est d'utiliser un nouveau tableau plus output (qui re�oit une copie de input)
	using wrapper = std::reference_wrapper<std::vector<unsigned>>;
	std::vector<unsigned> temp(input.size());
	wrapper T[2] = { wrapper(output), wrapper(temp) };
	std::vector<unsigned> predicate(input.size());
	std::copy(input.begin(), input.end(), output.begin());
	for(unsigned numeroBit=0; numeroBit<sizeof(unsigned)*8; ++numeroBit) 
	{

		const int ping = numeroBit & 1;
		const int pong = 1 - ping;
		fill_predicate(T[ping].get(),predicate,numeroBit);
		std::vector<unsigned>& src = T[ping].get();
		::partition(T[ping].get(),T[pong].get(),predicate);
	}
}

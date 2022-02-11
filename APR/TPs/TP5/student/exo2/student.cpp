#include <iostream>
#include <functional>
#include <exo2/student.h>
#include <algorithm>

namespace 
{
	
}

bool StudentWorkImpl::isImplemented() const {
	return false;
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
		// TODO (extraire le bit, partitionner)
	}
}

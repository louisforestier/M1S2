#pragma once

#include <OPP.h>
#include <StudentWork.h>
#include <vector>
#include <iostream>

#include <previous/transform.h>
#include <previous/inclusive_scan.h>
#include <previous/exclusive_scan.h>
#include <previous/scatter.h>

class StudentWorkImpl : public StudentWork
{
public:
	bool isImplemented() const;

	StudentWorkImpl() = default;
	StudentWorkImpl(const StudentWorkImpl &) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl &operator=(const StudentWorkImpl &) = default;

	template <typename T>
	void partition(std::vector<T> &input, std::vector<T> &output, std::vector<unsigned> &predicate)
	{
		//TODO: utiliser les transforms iterator pour eviter le map sur le predicat
		// - utiliser les reverse iterator pour faire le scan inclusif à l'envers sans faire reverse avant
		unsigned j = 0;
		std::vector<unsigned> head_position(predicate.size());
		std::vector<unsigned> tail_position(predicate.size());
		std::vector<unsigned> not_predicate(predicate.size());
		std::vector<unsigned> map(predicate.size());
		auto transformIterator =
			OPP::make_transform_iterator(
				OPP::CountingIterator(0l),
				std::function([](long a) -> int
							  { return a; }));
		int n = predicate.size();
		
		OPP::transform(predicate.begin(), predicate.end(), not_predicate.begin(), [](unsigned u){ return !u; });
		OPP::exclusive_scan(not_predicate.begin(), not_predicate.end(), head_position.begin(), std::plus<>(), unsigned(0));
		//ne fonctionne pas avec tail_position.rbegin()
		OPP::inclusive_scan(predicate.rbegin(), predicate.rend(), tail_position.begin(), std::plus<>());
		std::reverse(tail_position.begin(), tail_position.end());
		OPP::transform(
			transformIterator + 0,
			transformIterator + n,
			map.begin(),
			std::function(
				[&head_position, &tail_position, &predicate, n](int a)
				{
					if (predicate[a])
					{
						return n - tail_position[a];
					}
					return head_position[a];
				}));
		OPP::scatter(input.begin(),input.end(),map.begin(),output.begin());
 	}

template <typename T>
void run_radixSort_parallel(
	std::vector<T> &input,
	std::vector<T> &output)
{
	// TODO
	using wrapper = std::reference_wrapper<std::vector<T>>;
	std::vector<T> temp(input.size());
	wrapper W[2] = {wrapper(output), wrapper(temp)};
	std::vector<unsigned> predicate(input.size());
	std::copy(input.begin(), input.end(), output.begin());
	for (unsigned numeroBit = 0; numeroBit < sizeof(T) * 8; ++numeroBit)
	{

		const int ping = numeroBit & 1;
		const int pong = 1 - ping;
		std::vector<T> &src = W[ping].get();
		OPP::transform(src.begin(), src.end(), predicate.begin(),
					   [numeroBit](T value) -> unsigned
					   {
						   if (numeroBit == 0)
						   {
							   return (value & 0x1);
						   }
						   return ((value >> numeroBit) & 0x1);
					   });
		this->partition(W[ping].get(), W[pong].get(), predicate);
	}
}

// Illustration de l'utilisation des it�rateurs "counting" et "transform" d�finis dans OPP.h
// Ils sont encore exp�rimental, mais bon ils font le boulot ;-)
// TODO : acheter un antihistaminique
void check();
};
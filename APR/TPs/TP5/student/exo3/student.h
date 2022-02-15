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
	inline unsigned extract(T value, unsigned bit)
	{
		if (bit == 0)
		{
			return 1 - (value & 0x1);
		}
		return 1 - ((value >> bit) & 0x1);
	}

	template <typename T>
	void partition(std::vector<T> &input, std::vector<T> &output, std::vector<unsigned> &predicate)
	{
		unsigned j = 0;
		std::vector<unsigned> head_position(predicate.size());
		std::vector<unsigned> tail_position(predicate.size());
		std::vector<unsigned> not_predicate(predicate.size());
		std::vector<unsigned> map(predicate.size());
		OPP::transform(predicate.begin(), predicate.end(), not_predicate.begin(), [](unsigned u)
					   { return !u; });
		OPP::exclusive_scan(not_predicate.begin(), not_predicate.end(), head_position.begin(), std::plus<>(), unsigned(0));
		std::vector<unsigned> reverse_predicate(predicate.size());
		std::copy(predicate.begin(), predicate.end(), reverse_predicate.begin());
		std::reverse(reverse_predicate.begin(), reverse_predicate.end());
		OPP::inclusive_scan(reverse_predicate.begin(), reverse_predicate.end(), tail_position.begin(), std::plus<>());
		std::reverse(tail_position.begin(), tail_position.end());
		auto transformIterator =
			OPP::make_transform_iterator(
				OPP::CountingIterator(0l),
				std::function([](long a) -> int
							  { return a; }));
		int n = predicate.size();
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
	using wrapper = std::reference_wrapper<std::vector<unsigned>>;
	std::vector<unsigned> temp(input.size());
	wrapper W[2] = {wrapper(output), wrapper(temp)};
	std::vector<unsigned> predicate(input.size());
	std::copy(input.begin(), input.end(), output.begin());
	for (unsigned numeroBit = 0; numeroBit < sizeof(unsigned) * 8; ++numeroBit)
	{

		const int ping = numeroBit & 1;
		const int pong = 1 - ping;
		std::vector<unsigned> nbits(predicate.size());
		std::fill(nbits.begin(), nbits.end(), numeroBit);
		std::vector<unsigned> &src = W[ping].get();
		OPP::transform(src.begin(), src.end(), nbits.begin(), predicate.begin(),
					   [](T value, unsigned bit) -> unsigned
					   {
						   if (bit == 0)
						   {
							   return (value & 0x1);
						   }
						   return ((value >> bit) & 0x1);
					   });
		this->partition(W[ping].get(), W[pong].get(), predicate);
	}
}

// Illustration de l'utilisation des it�rateurs "counting" et "transform" d�finis dans OPP.h
// Ils sont encore exp�rimental, mais bon ils font le boulot ;-)
// TODO : acheter un antihistaminique
void check();
};
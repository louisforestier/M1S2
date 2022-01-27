#pragma once
#include <execution>
#include <algorithm>
#include <StudentWork.h>
#include <vector>

class StudentWorkImpl: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWorkImpl() = default;
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	template<typename T>
	T run_sum(const std::vector<T>& input)
	{
		T res = std::reduce(std::execution::par_unseq, input.begin(), input.end(),T(0), std::plus<>());
		return res;
	}

	template<typename T>
	T run_sum_square(const std::vector<T>& input)
	{
		std::vector<T> tmp=input;
		std::transform(std::execution::par_unseq ,input.begin(),input.end(),tmp.begin(),
			[](T i) -> T {return i*i;});
		T res = std::reduce(std::execution::par_unseq, tmp.begin(), tmp.end(),T(0), std::plus<>());
		return res;
	}

	template<typename T>
	T run_sum_square_opt(const std::vector<T>& input)
	{
		T res = std::transform_reduce(std::execution::par_unseq, input.begin(),input.end(),T(0),std::plus<>(),
			[](T i) -> T {return i*i;});
		return res;
	}

};
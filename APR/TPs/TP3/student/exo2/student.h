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
		// TODO: parallel sum
		return T(0);
	}
	
	template<typename T>
	T run_sum_square(const std::vector<T>& input)
	{
		// TODO: parallel square and then sum
		return T(0);
	}

	template<typename T>
	T run_sum_square_opt(const std::vector<T>& input)
	{
		// TODO: parallel square and sum in once
		return T(0);
	}

};
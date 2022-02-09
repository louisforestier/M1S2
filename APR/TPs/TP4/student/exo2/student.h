#pragma once

#include <StudentWork.h>
#include <vector>
#include <functional>
#include "exclusive_scan.h"

class StudentWorkImpl: public StudentWork
{
public:

	bool isImplemented() const {
		return true;
	}

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	template< typename T>
	void run_scan_sequential(
		std::vector<T>& input,
		std::vector<T>& output,
		const T& Tinit,
		std::function<T(T,T)>& functor
	) {		
		output[0] = Tinit;
		for (auto iter = input.begin(); iter < input.end()-1; iter++)
		{
			output[iter-input.begin()+1] = functor(output[iter-input.begin()],*iter);
		}
	}
	
	template< typename T>
	void run_scan_parallel(
		std::vector<T>& input,
		std::vector<T>& output,
		const T& Tinit,
		std::function<T(T,T)>& functor
	) {		
		OPP::exclusive_scan(
			input.begin(), 
			input.end(), 
			output.begin(), 
			std::forward<std::function<T(T,T)>>(functor),
			Tinit
		);
	}
};
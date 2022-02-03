#pragma once

#include <StudentWork.h>
#include <vector>
#include <functional>
#include <utility>
#include "inclusive_scan.h"

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
		std::function<T(T,T)>& functor
	) {		
		// TODO
	}
	
	template< typename T>
	void run_scan_parallel(
		std::vector<T>& input,
		std::vector<T>& output,
		std::function<T(T,T)>& functor
	) {		
		OPP::inclusive_scan(
			input.begin(), 
			input.end(), 
			output.begin(), 
			std::forward<std::function<T(T,T)>>(functor)
		);
	}
};
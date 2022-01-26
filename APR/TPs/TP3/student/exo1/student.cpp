#include <execution>
#include <algorithm>
#include <iostream>
#include <exo1/student.h>

namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return false;
}

void StudentWorkImpl::run_square(const std::vector<int>& input, std::vector<int>& output) 
{
	// TODO use the std::transform (aka MAP) pattern in parallel mode
	// to do something like ....
	//for(auto i=input.size() ; i--;)
	//	output[i] = input[i]*input[i];
}

void StudentWorkImpl::run_sum(
	const std::vector<int>& input_a,
	const std::vector<int>& input_b,
	std::vector<int>& output
) {
	// TODO: parallel sum using std::transform
}

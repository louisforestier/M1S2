#include <iostream>
#include <exo3/student.h>

#include <exo3/transform.h>

namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_square(const std::vector<int>& input, std::vector<int>& output) 
{
	// TODO use the OPP:transform (aka MAP) pattern in parallel mode
	OPP::transform(input.begin(),input.end(),output.begin(), [](int i) -> int {return i*i;});
}

void StudentWorkImpl::run_sum(
	const std::vector<int>& input_a,
	const std::vector<int>& input_b,
	std::vector<int>& output
) {
	// TODO: parallel sum using OPP:transform
	OPP::transform(input_a.begin(),input_a.end(),input_b.begin(),output.begin(), [](int i, int j) -> int {return i+j;});
}

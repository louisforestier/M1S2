#include <iostream>
#include <exo1/student.h>


namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_partition_sequential(
	std::vector<int>& input,
	std::vector<int>& predicate,
	std::vector<int>& output
) {
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

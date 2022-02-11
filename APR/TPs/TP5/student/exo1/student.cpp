#include <iostream>
#include <exo1/student.h>


namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return false;
}

void StudentWorkImpl::run_partition_sequential(
	std::vector<int>& input,
	std::vector<int>& predicate,
	std::vector<int>& output
) {
	// TODO	
	std::vector<int> head_position(input.size());
	std::vector<int> tail_position(input.size());
	head_position[0] = 0 ;
	for ( int i = 0 ; i < input.size() ; i++){
		if (predicate[i])
		{
			head_position[i+1] = 1;
			tail_position[i] = 0;
		} 
		else
		{
			head_position[i+1] = 0;
			tail_position[i] = 1;
		}
		int j = 1;		
		
	}
}

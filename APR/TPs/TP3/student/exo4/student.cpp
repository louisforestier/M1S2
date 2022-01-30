#include <iostream>
#include <exo4/student.h>

#include <exo4/gather.h>
#include <exo4/scatter.h>

namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_gather(
	const std::vector<int>& input,
	const std::vector<size_t>& map,
	std::vector<int>& output
) {
	std::cout<<"input"<<std::endl;
	for (auto i : input)
	{
		std::cout<<i <<";";
	}
	std::cout<<std::endl;
	std::cout<<"map"<<std::endl;
	for (auto i : map)
	{
		std::cout<<i <<";";
	}
	std::cout<<std::endl;

	OPP::gather(input.begin(), input.end(), map.begin(), output.begin());
	std::cout<<"output"<<std::endl;
	for (auto i : output)
	{
		std::cout<<i <<";";
	}
	std::cout<<std::endl;

}

void StudentWorkImpl::run_scatter(
	const std::vector<int>& input,
	const std::vector<size_t>& map,
	std::vector<int>& output
) {
	std::cout<<"input"<<std::endl;
	for (auto i : input)
	{
		std::cout<<i <<";";
	}
	std::cout<<std::endl;
	std::cout<<"map"<<std::endl;
	for (auto i : map)
	{
		std::cout<<i <<";";
	}
	std::cout<<std::endl;
	
	OPP::scatter(input.begin(), input.end(), map.begin(), output.begin());
	std::cout<<"output"<<std::endl;
	for (auto i : output)
	{
		std::cout<<i <<";";
	}
	std::cout<<std::endl;

}

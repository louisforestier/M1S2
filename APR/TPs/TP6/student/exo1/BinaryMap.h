#pragma once
#include <OPP_cuda.cuh>

namespace
{
	// TODO: add a kernel, fill the following method
	template<typename T, typename Functor>
	void BinaryMap(
		OPP::CUDA::DeviceBuffer<int>& dev_a,
		OPP::CUDA::DeviceBuffer<int>& dev_b,
		OPP::CUDA::DeviceBuffer<int>& dev_result,
		Functor& functor
	) {
	}
}
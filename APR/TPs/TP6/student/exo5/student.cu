#include <iostream>
#include <exo5/student.h>
#include <OPP_cuda.cuh>

namespace 
{
	// Vous utiliserez ici les types uchar3 et float3 (internet : CUDA uchar3)
	// Addition de deux "float3"
	__device__ 
	float3 operator+(const float3 &a, const float3 &b) 
	{
		return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
	}

	// TODO
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_filter(
	OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
	OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
	OPP::CUDA::DeviceBuffer<float>& dev_filter,
	const unsigned imageWidth, 
	const unsigned imageHeight,
	const unsigned filterWidth
) {
	// TODO
}

#include <iostream>
#include <exo4/student.h>
#include <exo3/mapFunctor.h>
#include <OPP_cuda.cuh>

namespace 
{
	// TODO	
	template<typename T, typename Functor>
	__host__
	void Scatter(
		OPP::CUDA::DeviceBuffer<T>& dev_input,
		OPP::CUDA::DeviceBuffer<T>& dev_output,
		Functor& map
	) {
		
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_thumbnail_scatter(
	OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
	OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
	OPP::CUDA::DeviceBuffer<uchar2>& dev_map,
	const unsigned imageWidth, 
	const unsigned imageHeight
) {
	::MapFunctor<3> map(
		dev_map.getDevicePointer(),
		imageWidth,
		imageHeight
	);

	::Scatter<uchar3,MapFunctor<3>>(
		dev_inputImage, dev_outputImage, map
	);
}

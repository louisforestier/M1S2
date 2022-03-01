#include <iostream>
#include <exo3/student.h>
#include <OPP_cuda.cuh>
#include <exo3/mapFunctor.h>

namespace 
{
	// TODO: fill this host functor to do a Gather onto device ...
	template<typename T, typename Functor>
	__host__
	void Gather(
		OPP::CUDA::DeviceBuffer<T>& dev_input,
		OPP::CUDA::DeviceBuffer<T>& dev_output,
		Functor& map
	) {
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_thumbnail_gather(
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

	::Gather<uchar3,MapFunctor<3>>(
		dev_inputImage, dev_outputImage, map
	);
}

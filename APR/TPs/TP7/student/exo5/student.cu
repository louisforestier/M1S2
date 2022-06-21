#include <iostream>
#include <exo5/student.h>
#include <OPP_cuda.cuh>

using uchar = unsigned char;

namespace 
{
	// idem exo3
	template<int NB_WARPS>
	__device__ 
	__forceinline__
	void loadSharedMemoryCommutative(float const*const data) 
	{
		// TODO
	}

	// idem exo4
	__device__ 
	__forceinline__
	void reduceJumpingStep(const int jump)
	{
		// TODO
	}

	// nouvelle fonction !
	__device__ 
	__forceinline__
	void reduceLastWarp()
	{
		// attention au mot clé volatile ... essentiel !
		volatile float*const shared = OPP::CUDA::getSharedMemory<float>();
		const auto tid = threadIdx.x;
		if( tid < 32 ) 
		{
			// TODO
		}
		__syncthreads();
		
	}
	
	// 
	template<int NB_WARPS>
	__device__
	__forceinline__
	float reducePerBlock(
		float const*const source
	) {
		// TODO
	}

	
	// idem exo3
	template<int NB_WARPS>
	__device__
	__forceinline__
	void fillBlock(
		const float color, 
		float*const result
	) {
		// TODO
	}


	// idem exo1
	template<int NB_WARPS>
	struct EvaluateWarpNumber {
		enum { res = 1 };
	};
	template<>
	struct EvaluateWarpNumber<1> {
		enum { res = 16 };
	};
	template<>
	struct EvaluateWarpNumber<2> {
		enum { res = 8 };
	};
	template<>
	struct EvaluateWarpNumber<4> {
		enum { res = 4 };
	};
	template<>
	struct EvaluateWarpNumber<8> {
		enum { res = 4 };
	};
	template<>
	struct EvaluateWarpNumber<16> {
		enum { res = 2 };
	};
	template<int NB_WARPS=32>
	__global__
	__launch_bounds__(32*NB_WARPS , EvaluateWarpNumber<NB_WARPS>::res)
	void blockEffectKernel( 
		float const*const source, 
		float *const result
	) {
		const float sumInBlock = reducePerBlock<NB_WARPS>(source);
		fillBlock<NB_WARPS>(sumInBlock, result);
	}
}


// Attention : ici la taille des vecteurs n'est pas toujours un multiple du nombre de threads !
// Il faut donc corriger l'exemple du cours ...
void StudentWorkImpl::run_blockEffect(
	OPP::CUDA::DeviceBuffer<float>& dev_source,
	OPP::CUDA::DeviceBuffer<float>& dev_result,
	const unsigned nbWarps
) {
	const auto size = dev_source.getNbElements();
	dim3 threads(32*nbWarps);
	dim3 blocks((size + 1023) / 1024);
	const size_t sizeSharedMemory(threads.x*sizeof(float));

	switch(nbWarps) {
		case 1:
			::blockEffectKernel<1> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 2:
			::blockEffectKernel<2> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 4:
			::blockEffectKernel<4> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 8:
			::blockEffectKernel<8> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 16:
			::blockEffectKernel<16> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 32:
			::blockEffectKernel<32><<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		default:
			::blockEffectKernel<32><<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
	}
}

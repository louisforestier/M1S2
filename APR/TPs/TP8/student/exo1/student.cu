#include <iostream>
#include <exo1/student.h>
#include <OPP_cuda.cuh>

#ifndef WIN32
# include <cfloat>
#endif

namespace 
{
	__device__
	float3 RGB2HSV( const uchar3 inRGB ) {
		const float R = float( inRGB.x ) / 256.f;
		const float G = float( inRGB.y ) / 256.f;
		const float B = float( inRGB.z ) / 256.f;

		const float min		= fminf( R, fminf( G, B ) );
		const float max		= fmaxf( R, fmaxf( G, B ) );
		const float delta	= max - min;

		// H
		float H;
		if		( delta < FLT_EPSILON )  
			H = 0.f;
		else if	( max == R )	
			H = 60.f * ( G - B ) / ( delta + FLT_EPSILON )+ 360.f;
		else if ( max == G )	
			H = 60.f * ( B - R ) / ( delta + FLT_EPSILON ) + 120.f;
		else					
			H = 60.f * ( R - G ) / ( delta + FLT_EPSILON ) + 240.f;
		while	( H >= 360.f )	
			H -= 360.f ;

		// S
		const float S = max < FLT_EPSILON ? 0.f : 1.f - min / max;

		// V
		const float V = max;

		return make_float3( H, S, V );
	}

	__device__
	uchar3 HSV2RGB( const float H, const float S, const float V ) {
		const float	d	= H / 60.f;
		const int	hi	= int(d) % 6;
		const float f	= d - float(hi);

		const float l   = V * ( 1.f - S );
		const float m	= V * ( 1.f - f * S );
		const float n	= V * ( 1.f - ( 1.f - f ) * S );

		float R, G, B;

		if		( hi == 0 ) 
			{ R = V; G = n;	B = l; }
		else if ( hi == 1 ) 
			{ R = m; G = V;	B = l; }
		else if ( hi == 2 ) 
			{ R = l; G = V;	B = n; }
		else if ( hi == 3 ) 
			{ R = l; G = m;	B = V; }
		else if ( hi == 4 ) 
			{ R = n; G = l;	B = V; }
		else				
			{ R = V; G = l;	B = m; }
			
		return make_uchar3( R * 256.f, G * 256.f, B * 256.f );
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_RGB2HSV(
	OPP::CUDA::DeviceBuffer<uchar3>& dev_source,
	OPP::CUDA::DeviceBuffer<float>& dev_Hue,
	OPP::CUDA::DeviceBuffer<float>& dev_Saturation,
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	const unsigned width,
	const unsigned height
) {
	// TODO
}

void StudentWorkImpl::run_HSV2RGB(
	OPP::CUDA::DeviceBuffer<float>& dev_Hue,
	OPP::CUDA::DeviceBuffer<float>& dev_Saturation,
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	OPP::CUDA::DeviceBuffer<uchar3>& dev_result,
	const unsigned width,
	const unsigned height
) {
	// TODO
}

#include <immintrin.h> //AVX+SSE Extensions#include <vector>
#include <cmath>
#include <iostream>
#include <exo4/student.h>

namespace {
	// TODO: add your local classes/functions here
	struct Convertor 
	{ 
		union {
			__m256 avx; 
			float f[8];
		} u;
		// constructor
		Convertor(const __m256& m256) { u.avx = m256; };
		// accessor to element i (between 0 and 7 included)
		float operator()(int i) const 
		{ 			
			return u.f[i]; 
		}
		float& operator()(int i) 
		{ 			
			return u.f[i]; 
		}
		// prints data on a given stream
		friend std::ostream& operator<<(std::ostream&, const Convertor&c);
	};

	std::ostream& operator<<(std::ostream& os, const Convertor&c) 
	{
		os << "{ ";
		for(int i=0; i<7; ++i) {
			os << c(i) << ", ";
		}
		return os << c(7) << " }";
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

// calculate with floats
#pragma optimize("", off)
float StudentWorkImpl::run(float const * const input, const size_t size) 
{
	// attention au sch�ma de calcul ... 
	// TODO
	float result=0;
	for (size_t i = 0; i < size; i++)
	{
		result+= input[i];
	}
	return result;
}

// calculate with mm256
#pragma optimize("", off)
float StudentWorkImpl::run(__m256 const *const input, const size_t size) 
{
	// attention au sch�ma de calcul ... 
	// TODO
	__m256 sum = input[0];
	for (size_t i = 1; i < size; i++)
	{
		sum =_mm256_add_ps(sum,input[i]);
	}
	Convertor c(sum);
	float res =0.f;
	for (int i = 0; i < 8; i++)
	{
		res += c(i);
	}
	

	return res;
}

#include <immintrin.h> //AVX+SSE Extensions#include <vector>
#include <cmath>
#include <iostream>
#include <exo5/student.h>

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
	return false;
}


#pragma optimize("", off)
void StudentWorkImpl::run(
	float const * const input_matrix, // square matrix, stored by row
	float const * const input_vector, 
	float       * const output_vector, 
	const size_t vector_size
) {
	// each coordinates of the result vector is a dot product 
	// between the row i of the input square matrix and the 
	// input vector ... 
	// TODO
}

#pragma optimize("", off)
void StudentWorkImpl::run(
	__m256 const * const input_matrix, // square matrix, stored by row
	__m256 const * const input_vector, 
	float        * const output_vector, 
	const size_t vector_size
) {
	// each coordinates of the result vector is a dot product 
	// between the row i of the input square matrix and the 
	// input vector ... 
	// 
	// NB: the matrix contains vector_size columns (with AVX), and so 8*vector_size rows ...
	//     for instance if vector_size is 1, you have 1 column, and 8 rows, so that the matrix
	//     contains 8x8 floats :-)
	// TODO
}
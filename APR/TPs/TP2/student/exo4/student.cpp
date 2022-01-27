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
	__m256 sum = input[0];
	for (size_t i = 1; i < size; i++)
	{
		sum =_mm256_add_ps(sum,input[i]);
	}
	//on découpe le m256 en 2 parties qu'on additionne horizontalement
	/*
	 1,2,3,4,5,6,7,8
	+5,6,7,8,1,2,3,4
	=6,8,10,12,6,8,10,12
	*/
	__m256 partie_basse = sum;
	__m256 partie_haute = _mm256_permute2f128_ps(sum,sum,1);
	__m256 somme_horizontale = _mm256_add_ps(partie_basse, partie_haute);
	//on fait additionner les nombres d'indice pair avec le nombre suivant 2 fois de suite
	/*
	 6,8,10,12,6,8,10,12
	 1ere fois : 14,22,14,22,14,22,14,22
	 2ème fois : 36,36,36,36,36,36,36,36
	*/
	somme_horizontale = _mm256_hadd_ps(somme_horizontale,somme_horizontale);
	somme_horizontale = _mm256_hadd_ps(somme_horizontale,somme_horizontale);
	Convertor c(somme_horizontale);
    return c(0);
}

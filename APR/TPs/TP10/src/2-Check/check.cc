#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>

bool check( const int N,
            const char*vectorFileName,
            const char*matrixFileName,
            const char*resultFileName )
{
  // open file, check the results
  const int dv = open(vectorFileName, O_RDONLY);
  if( dv == -1 ) {
    std::cerr<<"Unable to open "<<vectorFileName<<" ... "<<std::endl;
    return false;
  }
  const int dm = open(matrixFileName, O_RDONLY);
  if( dm == -1 ) {
    std::cerr<<"Unable to open "<<matrixFileName<<" ... "<<std::endl;
    close(dv);
    return false;
  }
  const int dr = open(resultFileName, O_RDONLY);
  if( dr == -1 ) {
    std::cerr<<"Unable to open "<<resultFileName<<" ... "<<std::endl;
    close(dv);
    close(dm);
    return false;
  }
  float *A = new float[N], *B = new float[N];
  const int size = sizeof(float)*N;
  if( read(dv, B, size) != size ) {
    std::cerr<<"vector file is too small ... unable to read N floats"<<std::endl;
    delete A;
    delete B;
    close(dv);
    close(dm);
    close(dr);
    return false;
  }
  for(int r=0; r<N; ++r) {
    if( read(dm, A, size) != size ) {
      std::cerr<<"matrix file is too small ... unable to get row "<<r<<std::endl;
      delete A;
      delete B;
      close(dv);
      close(dm);
      close(dr);
      return false;
    }
    float truth = 0.f;
    for(int c=0; c<N; ++c) truth += A[c]*B[c];
    float result;
    if( read(dr, &result, sizeof(float)) != sizeof(float)) {
      std::cerr<<"Unable to read result vector at position "<<r<<std::endl;
      delete A;
      delete B;
      close(dv);
      close(dm);
      close(dr);
      return false;
    }
    if( fabs(truth-result) > 1e-3f ) {
      std::cerr<<"Your calculation is far too different to the good result"<<std::endl;
      std::cerr<<" -> row="<<r<<", truth="<<truth<<", result="<<result<<std::endl;
      delete A;
      delete B;
      close(dv);
      close(dm);
      close(dr);
      return false;
    }
  }
  delete A;
  delete B;
  close(dv);
  close(dm);
  close(dr);
  return true;
}

int main(int argc, char**argv)
{
  int e = 10;
  const char*vectorFileName = "vector.bin";
  const char*resultFileName = "result.bin";
  const char*matrixFileName = "matrix.bin";
  for(int i=0; i<argc-1; ++i) {
    if( !strcmp("-e", argv[i]) ) {
      int value;
      if( sscanf(argv[i+1], "%d", &value) == 1 ) {
	       e = std::max(3, std::min(value,32));
	       i++;
      }
    }
    else if( !strcmp("-v", argv[i]) && argv[i+1][0] != '-' ) {
      vectorFileName = argv[++i];
    }
    else if( !strcmp("-m", argv[i]) && argv[i+1][0] != '-' ) {
      matrixFileName = argv[++i];
    }
    else if( !strcmp("-x", argv[i]) && argv[i+1][0] != '-' ) {
      resultFileName = argv[++i];
    }
    else if( !strcmp("-h", argv[i])) {
      std::cout<<"Usage:"<<std::endl;
      std::cout<<"\t-e <e>: log2(n), for vector of size n and matrix (n x n)"<<std::endl;
      std::cout<<"\t-v <f>: set the filename for the vector"<<std::endl;
      std::cout<<"\t-m <f>: set the filename for the matrix"<<std::endl;
      std::cout<<"\t-x <f>: set the filename for the resulting vector"<<std::endl;
    }
  }
  const int N = 1<<e;

  if( check(N, vectorFileName, matrixFileName, resultFileName) )
    std::cout<<"Well done, your software seems to be correct!"<<std::endl;
  else
    std::cout<<"Your result is uncorrect ... try again!"<<std::endl;

  return EXIT_SUCCESS;
}

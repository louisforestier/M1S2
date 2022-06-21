#include <mpi.h>
#include <DistributedRowMatrix.h>
#include <DistributedBlockVector.h>
#include <2-Produit/Produit.h>
#include <cstring>
// 
/* Effectuer le calcul X = A fois B ... */
void produit(
    const OPP::MPI::Communicator& communicator,
    const DistributedRowMatrix &A,
    const DistributedBlockVector &B,
    DistributedBlockVector &X,
    const int N
){
  OPP::MPI::Ring ring(communicator);
  int p = ring.getSize();
  int q = ring.getRank();
  int r = N / p;

  float *sendArray = new float[r];
  float *recvArray = new float[r];
  
  for (int i = X.Start(); i < X.End(); i++) 
    X[i] = 0;

  int k = 0;
  for (int i = B.Start(); i < B.End(); i++) 
    sendArray[k++] = B[i];

  for (int step = 0; step < p; ++step) {
    MPI_Request requests[2];
    requests[0] = ring.AsyncSend(sendArray, r, MPI_FLOAT);
    requests[1] = ring.AsyncRecv(recvArray, r, MPI_FLOAT);

    for (int j = 0; j < r; ++j) {
      for (int i = 0; i < r; ++i) {
        int column = i + ((q + p - step) % p) * r;
        X[X.Start() + j] += A[A.Start() + j][column] * sendArray[i];
      }
    }
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    std::swap(sendArray,recvArray);
  }
  delete sendArray;
  delete recvArray;
}
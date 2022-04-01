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
    // TODO
}

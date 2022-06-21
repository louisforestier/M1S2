#include <iostream>
#include <cstring>
#include <algorithm>
#include <3-2-broadcast/Broadcast.h>
#include <MPI/OPP_MPI.h>

// Version pipeline ...
void Broadcast(
    const int k, // numéro du processeur émetteur, dans 0..P-1
    int *const addr, // pointeur sur les données à envoyer/recevoir
    const int N, // nombre d'entiers à envoyer/recevoir
    const int M // taille d'un paquet de données ...
) {  
    OPP::MPI::Ring ring(MPI_COMM_WORLD);
    int rank = ring.getRank();
    int p = ring.getSize();
    int r = N/M;
    if (p <= 1)
        return; 
    if( rank == k )
        for(int i=0; i<r; ++i)
            ring.Send(addr+(i*M), N/r, MPI_INT);
    else if( ((rank+1)%p) == k)
        for(int i = 0; i<r; ++i)
            ring.Recv(addr+(i*M), N/r,MPI_INT);
    else {
        ring.Recv(addr, N/r, MPI_INT);
        for(int i=0; i<r-1; ++i)
        {
            MPI_Request h = ring.AsyncSend(addr+(i*M), N/r,MPI_INT);
            MPI_Status status;
            ring.Recv(addr+((i+1)*M), N/r,MPI_INT);
            MPI_Wait(&h, &status);
        }
        ring.Send(addr+((r-1)*M), N/r,MPI_INT);
    }
}

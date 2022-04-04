#include <mpi.h>
#include <1-Resolution/Resolution.h>
#include <algorithm>
#include <vector>

void Solve(
    const OPP::MPI::Communicator& communicator,
    const DistributedRowMatrix& L,
    const DistributedBlockVector& B,
    DistributedBlockVector& X,
    const int N
) {
    OPP::MPI::Ring ring(communicator);
    
    // Here, we have a block of row (take care to the distribution!)
    // block size ... or B.End() - B.Start() except the last processor (it can be smaller for last block)
    const int m = (N+ring.getSize()-1) / ring.getSize(); 
    // check it is ok
    if( m < B.End() - B.Start() )
        std::cerr << "Bad value for m="<<m << std::endl;
        
    // TODO
    int rank = ring.getRank();
    int p = ring.getSize();
    std::vector<float> tmp;
    for (int i = B.Start(); i < B.End(); i++)
    {
        tmp.push_back(B[i]);
    }
    
    //std::copy(B.Start(),B.End(),tmp);

    for (int col = 0; col < B.End(); col++)
    {
        if (rank < col/m)
        {
            X[col] = tmp[col-B.Start()] / L[col][col];
            ring.Send(&X[col], 1, MPI_FLOAT);
            for (int line = col; line < N; line++)
            {
                ring.Recv(&X[col], 1, MPI_FLOAT);
                tmp[line-B.Start()] -= L[line][col] * X[col];
            }
        }        
    }
}
/*
  float* newB = new float[B.End() - B.Start()];

  for (int i = B.Start(), n = 0; i < B.End(); i++, n++) newB[n] = B[i];

  if (ring.getRank() == 0) {
    for (int col = 0; col < B.End(); ++col) {
      X[col] = newB[col] / L[col][col];

      for (int line = col; line < B.End(); ++line)
        newB[line] -= L[line][col] * X[col];
    }

    ring.Send(&X[0], m, MPI_FLOAT);
  } else {
    float* prevX = new float[m * ring.getRank()];

    for (int proc = 0; proc < ring.getRank(); proc++) {
      ring.Recv(prevX + m * proc, m, MPI_FLOAT);

      if (ring.getNext() != 0) ring.Send(prevX + m * proc, m, MPI_FLOAT);

      for (int col = proc * m; col < ((proc + 1) * m); col++) {
        for (int line = B.Start(); line < B.End(); ++line)
          newB[line - B.Start()] -= L[line][col] * prevX[col];
      }
    }

    for (int col = B.Start(); col < B.End(); col++) {
      X[col] = newB[col - B.Start()] / L[col][col];

      for (int line = col; line < B.End(); ++line)
        newB[line - B.Start()] -= L[line][col] * X[col];
    }

    if (ring.getNext() != 0) ring.Send(&X[m * ring.getRank()], m, MPI_FLOAT);
  }

*/
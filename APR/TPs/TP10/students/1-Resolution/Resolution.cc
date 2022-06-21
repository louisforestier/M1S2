#include <mpi.h>
#include <1-Resolution/Resolution.h>
#include <algorithm>
#include <vector>

void Solve(
    const OPP::MPI::Communicator &communicator,
    const DistributedRowMatrix &L,
    const DistributedBlockVector &B,
    DistributedBlockVector &X,
    const int N)
{
  OPP::MPI::Ring ring(communicator);

  // Here, we have a block of row (take care to the distribution!)
  // block size ... or B.End() - B.Start() except the last processor (it can be smaller for last block)
  const int m = (N + ring.getSize() - 1) / ring.getSize();
  // check it is ok
  if (m < B.End() - B.Start())
    std::cerr << "Bad value for m=" << m << std::endl;

  int rank = ring.getRank();
  int p = ring.getSize();
  float *tmp = new float[m];
  int n = 0;
  for (int i = 0; i < m; i++)
    tmp[i] = B[i + rank * m];

  if (rank == 0)
  {
    for (int col = 0; col < m; col++)
    {
      {
        X[col] = tmp[col] / L[col][col];
        for (int row = col; row < m; row++)
        {
          tmp[row] -= L[row][col] * X[col];
        }
      }
    }
    ring.Send(&X[0], m, MPI_FLOAT);
  }
  else if ((rank + 1) % p == 0)
  {
    float *recvBuffer = new float[rank * m];
    for (int i = 0; i < rank; i++)
    {
      ring.Recv(recvBuffer + m * i, m, MPI_FLOAT);

      for (int col = i * m; col < (i + 1) * m; col++)
      {
        for (int row = rank * m; row < (rank + 1) * m; ++row)
        {
          tmp[row % m] -= L[row][col] * recvBuffer[col];
        }
      }
    }
    for (int col = rank * m; col < N; col++)
    {
      X[col] = tmp[col % m] / L[col][col];

      for (int row = col; row < N; ++row)
        tmp[row % m] -= L[row][col] * X[col];
    }
    delete recvBuffer;
  }
  else
  {
    float *recvBuffer = new float[rank * m];
    for (int i = 0; i < rank; i++)
    {
      ring.Recv(recvBuffer + m * i, m, MPI_FLOAT);

      ring.Send(recvBuffer + m * i, m, MPI_FLOAT);

      for (int col = i * m; col < (i + 1) * m; col++)
      {
        for (int row = rank * m; row < (rank + 1) * m; ++row)
          tmp[row % m] -= L[row][col] * recvBuffer[col];
      }
    }

    for (int col = rank * m; col < (rank + 1) * m; col++)
    {
      X[col] = tmp[col % m] / L[col][col];

      for (int row = col; row < (rank + 1) * m; ++row)
        tmp[row % m] -= L[row][col] * X[col];
    }

    ring.Send(&X[rank * m], m, MPI_FLOAT);
    delete recvBuffer;
  }
  delete tmp;
}

/* Version avec envoie par données :

void Solve(
    const OPP::MPI::Communicator &communicator,
    const DistributedRowMatrix &L,
    const DistributedBlockVector &B,
    DistributedBlockVector &X,
    const int N)
{
  OPP::MPI::Ring ring(communicator);

  // Here, we have a block of row (take care to the distribution!)
  // block size ... or B.End() - B.Start() except the last processor (it can be smaller for last block)
  const int m = (N + ring.getSize() - 1) / ring.getSize();
  // check it is ok
  if (m < B.End() - B.Start())
    std::cerr << "Bad value for m=" << m << std::endl;

  int rank = ring.getRank();
  int p = ring.getSize();
  float *tmp = new float[m];
  int n = 0;
  for (int i = 0; i < m; i++)
    tmp[i] = B[i + rank * m];

  if (rank == 0)
  {
    for (int col = 0; col < m; col++)
    {
      X[col] = tmp[col] / L[col][col];
      ring.Send(&X[col], 1, MPI_FLOAT);
      for (int row = col; row < m; row++)
      {
        tmp[row] -= L[row][col] * X[col];
      }
    }
  }
  else if ((rank + 1) % p == 0)
  {
    float *recvBuffer = new float[rank * m];
    for (int i = 0; i < rank; i++)
    {
      for (int col = i * m; col < (i + 1) * m; col++)
      {
        ring.Recv(recvBuffer + col, 1, MPI_FLOAT);
        for (int row = rank * m; row < (rank + 1) * m; ++row)
        {
          tmp[row % m] -= L[row][col] * recvBuffer[col];
        }
      }
    }
    for (int col = rank * m; col < N; col++)
    {
      X[col] = tmp[col % m] / L[col][col];

      for (int row = col; row < N; ++row)
        tmp[row % m] -= L[row][col] * X[col];
    }
    delete recvBuffer;
  }
  else
  {
    float *recvBuffer = new float[rank * m];
    for (int i = 0; i < rank; i++)
    {
      for (int col = i * m; col < (i + 1) * m; col++)
      {
          ring.Recv(recvBuffer + col, 1, MPI_FLOAT);
          ring.Send(recvBuffer + col, 1, MPI_FLOAT);
        for (int row = rank * m; row < (rank + 1) * m; ++row)
        {
          tmp[row % m] -= L[row][col] * recvBuffer[col];
        }
      }
    }

    for (int col = rank * m; col < (rank + 1) * m; col++)
    {
      X[col] = tmp[col % m] / L[col][col];
      ring.Send(&X[col], 1, MPI_FLOAT);

      for (int row = col; row < (rank + 1) * m; ++row)
        tmp[row % m] -= L[row][col] * X[col];
    }

    delete recvBuffer;
  }
  delete tmp;
}

*/
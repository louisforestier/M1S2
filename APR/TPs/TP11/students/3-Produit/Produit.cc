#include <3-Produit/Produit.h>
#include <memory>
#include <MPI/OPP_MPI.h>

namespace
{
    void BroadcastRow(const OPP::MPI::Torus& torus, int i, int j, std::shared_ptr<float> &srcAddr, std::shared_ptr<float>& destAddr, int L )
    {
        using Direction = OPP::MPI::Torus::Direction;
        if (torus.getRowRing().getRank() == j )
        {
            torus.Send(srcAddr.get(),L,MPI_FLOAT,Direction::EAST);
            for (int k = 0; k < L; k++)
                destAddr.get()[k] = srcAddr.get()[k];
        }
        else if (torus.getRowRing().getNext() == j)
        {
            torus.Recv(destAddr.get(),L,MPI_FLOAT,Direction::WEST);
        }
        else
        {
            torus.Recv(destAddr.get(),L,MPI_FLOAT,Direction::WEST);
            torus.Send(srcAddr.get(),L,MPI_FLOAT,Direction::EAST);
        }
    }
    
    void BroadcastCol(const OPP::MPI::Torus& torus, int i, int j, std::shared_ptr<float> &srcAddr, std::shared_ptr<float>& destAddr, int L )
    {
        using Direction = OPP::MPI::Torus::Direction;
        if (torus.getColumnRing().getRank() == i )
        {
            torus.Send(srcAddr.get(),L,MPI_FLOAT,Direction::SOUTH);
            for (int k = 0; k < L; k++)
                destAddr.get()[k] = srcAddr.get()[k];
        }
        else if (torus.getColumnRing().getNext() == i)
        {
            torus.Recv(destAddr.get(),L,MPI_FLOAT,Direction::NORTH);
        }
        else
        {
            torus.Recv(destAddr.get(),L,MPI_FLOAT,Direction::NORTH);
            torus.Send(srcAddr.get(),L,MPI_FLOAT,Direction::SOUTH);
        }
    }
    
} // namespace

void Produit(
    const OPP::MPI::Torus& torus,
    const DistributedBlockMatrix &A,
    const DistributedBlockMatrix &B,
          DistributedBlockMatrix &C
) {
    int n = torus.getRowRing().getSize();
    int i = torus.getRowRing().getRank();
    int j = torus.getColumnRing().getRank();

    int rows = A.End() - A.Start();
    int cols = A.End() - A.Start();

    std::shared_ptr<float> blockA(new float[rows*cols]);
    std::shared_ptr<float> blockB(new float[rows*cols]);
    std::shared_ptr<float> bufferA(new float[rows*cols]);
    std::shared_ptr<float> bufferB(new float[rows*cols]);
    for (int k = C.Start(); k < C.End(); k++)
    {
        for (int l = C[k].Start(); l < C[k].End(); l++)
        {
            auto I = k - C.Start();
            auto J = l - C[k].Start();
            blockA.get()[I * rows +J] = A[k][l];
            blockB.get()[I * rows +J] = B[k][l];
            C[k][l]= 0.0;
        }
    }
    for (int k = 0; k < n; k++)
    {
        BroadcastRow(torus,i,k,blockA,bufferA,rows*cols);
        BroadcastCol(torus,k,j,blockB,bufferB,rows*cols);
        for (int l = C.Start(); l < C.End(); l++)
        {
            for (int m = C[l].Start(); m < C[l].End(); m++)
            {
                auto I = l - C.Start();
                auto J = m - C[l].Start();
                for (int n = 0; n < rows; n++)
                    C[l][m]+=bufferA.get()[I * rows + k] * bufferB.get()[J+rows *k];
            }
        }
    }
}

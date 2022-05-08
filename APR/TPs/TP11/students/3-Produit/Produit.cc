#include <3-Produit/Produit.h>
#include <memory>
#include <MPI/OPP_MPI.h>

namespace
{
    void BroadcastRow(const OPP::MPI::Torus& torus, int i, int j, std::shared_ptr<float> &srcAddr, std::shared_ptr<float>& destAddr, int L )
    {
        using Direction = OPP::MPI::BidirRing::Direction;
        if (torus.getRowRing().getRank() == j )
        {
            torus.getRowRing().Send(srcAddr.get(),L,MPI_FLOAT,Direction::NEXT);
            for (int k = 0; k < L; k++)
                destAddr.get()[k] = srcAddr.get()[k];
        }
        else if (torus.getRowRing().getNext() == j)
        {
            torus.getRowRing().Recv(destAddr.get(),L,MPI_FLOAT,Direction::PREVIOUS);
        }
        else
        {
            torus.getRowRing().Recv(destAddr.get(),L,MPI_FLOAT,Direction::PREVIOUS);
            torus.getRowRing().Send(srcAddr.get(),L,MPI_FLOAT,Direction::NEXT);
        }
        
        
    }
    
    void BroadcastCol(const OPP::MPI::Torus& torus, int i, int j, std::shared_ptr<float> &srcAddr, std::shared_ptr<float>& destAddr, int L )
    {
        using Direction = OPP::MPI::BidirRing::Direction;
        if (torus.getColumnRing().getRank() == i )
        {
            torus.getColumnRing().Send(srcAddr.get(),L,MPI_FLOAT,Direction::NEXT);
            for (int k = 0; k < L; k++)
                destAddr.get()[k] = srcAddr.get()[k];
        }
        else if (torus.getColumnRing().getNext() == i)
        {
            torus.getColumnRing().Recv(destAddr.get(),L,MPI_FLOAT,Direction::PREVIOUS);
        }
        else
        {
            torus.getColumnRing().Recv(destAddr.get(),L,MPI_FLOAT,Direction::PREVIOUS);
            torus.getColumnRing().Send(srcAddr.get(),L,MPI_FLOAT,Direction::NEXT);
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

    int rows = A.m_m;
    int cols = A.m_n;

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
            bufferA.get()[I * rows +J] = 0.0;
            blockB.get()[I * rows +J] = B[k][l];
            bufferB.get()[I * rows +J] = 0.0;
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
                for (int n = 0; n < rows/2; n++)
                    C[l][m]+=bufferA.get()[I * rows + k] * bufferB.get()[J+rows *k];
            }
        }
    }
}

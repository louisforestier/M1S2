#include <4-Cannon/Produit.h>
#include <memory>

namespace
{
    void RotationHorizontale(const OPP::MPI::Torus& torus,std::shared_ptr<float> &srcAddr,std::shared_ptr<float> &destAddr,int L)
    {
        using Direction = OPP::MPI::BidirRing::Direction;
        torus.getRowRing().Send(srcAddr.get(),L,MPI_FLOAT,Direction::NEXT);
        torus.getRowRing().Recv(destAddr.get(),L,MPI_FLOAT,Direction::PREVIOUS);
    }

    void RotationVerticale(const OPP::MPI::Torus& torus,std::shared_ptr<float> &srcAddr,std::shared_ptr<float> &destAddr, int L)
    {
        using Direction = OPP::MPI::BidirRing::Direction;
        torus.getColumnRing().Send(srcAddr.get(),L,MPI_FLOAT,Direction::NEXT);
        torus.getColumnRing().Recv(destAddr.get(),L,MPI_FLOAT,Direction::PREVIOUS);
    }
} // namespace

void Produit(
    const OPP::MPI::Torus& torus,
    const DistributedBlockMatrix &A,
    const DistributedBlockMatrix &B,
          DistributedBlockMatrix &C
) {
    // TODO
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
            blockB.get()[I * rows +J] = B[k][l];
            C[k][l]= 0.0;
        }
    }
    RotationHorizontale(torus,blockA,bufferA,rows*cols);
    RotationVerticale(torus,blockB,bufferB,rows*cols);

    for (int k = 0; k < n; k++)
    {
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
        RotationHorizontale(torus,blockA,bufferA,rows*cols);
        RotationVerticale(torus,blockB,bufferB,rows*cols);
    }
}

#include <4-Cannon/Produit.h>
#include <memory>

namespace
{
    void RotationHorizontale(const OPP::MPI::Torus& torus,std::shared_ptr<float> &buffer,int L)
    {
        using Direction = OPP::MPI::BidirRing::Direction;
        torus.getRowRing().Send(buffer.get(),L,MPI_FLOAT,Direction::NEXT);
        torus.getRowRing().Recv(buffer.get(),L,MPI_FLOAT,Direction::PREVIOUS);
    }

    void RotationVerticale(const OPP::MPI::Torus& torus,std::shared_ptr<float> &buffer, int L)
    {
        using Direction = OPP::MPI::BidirRing::Direction;
        torus.getColumnRing().Send(buffer.get(),L,MPI_FLOAT,Direction::NEXT);
        torus.getColumnRing().Recv(buffer.get(),L,MPI_FLOAT,Direction::PREVIOUS);
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

    int rows = A.End() - A.Start();
    int cols = A.End() - A.Start();

    std::shared_ptr<float> bufferA(new float[rows*cols]);
    std::shared_ptr<float> bufferB(new float[rows*cols]);
    for (int i = C.Start(); i < C.End(); i++)
    {
        for (int j = C[i].Start(); j < C[i].End(); j++)
        {
            auto I = i - C.Start();
            auto J = j - C[i].Start();
            bufferA.get()[I * rows +J] = A[i][j];
            bufferB.get()[I * rows +J] = B[i][j];
            C[i][j]= 0.0;
        }
    }
    RotationHorizontale(torus,bufferA,rows*cols);
    RotationVerticale(torus,bufferB,rows*cols);

    for (int k = 0; k < n; k++)
    {
        for (int i = C.Start(); i < C.End(); i++)
        {
            for (int j = C[i].Start(); j < C[i].End(); j++)
            {
                auto I = i - C.Start();
                auto J = j - C[i].Start();
                for (int n = 0; n < rows; n++)
                    C[i][j]+=bufferA.get()[I * rows + k] * bufferB.get()[J+rows *k];
            }
        }
        RotationHorizontale(torus,bufferA,rows*cols);
        RotationVerticale(torus,bufferB,rows*cols);
    }
}

#include <memory>
#include <5-Fox/Produit.h>

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

    void RotationVerticale(const OPP::MPI::Torus& torus,std::shared_ptr<float> &buffer, int L)
    {
        using Direction = OPP::MPI::Torus::Direction;
        torus.Send(buffer.get(),L,MPI_FLOAT,Direction::SOUTH);
        torus.Recv(buffer.get(),L,MPI_FLOAT,Direction::NORTH);
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

    int rows = A.End() - A.Start();
    int cols = A.End() - A.Start();

    std::shared_ptr<float> blockA(new float[rows*cols]);
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
            bufferB.get()[I * rows +J] = B[k][l];
            C[k][l]= 0.0;
        }
    }
    for (int k = 0; k < n; k++)
    {
        int diag = (i + k) % n;
        BroadcastRow(torus,i,diag,blockA,bufferA,rows*cols);
        for (int j = C.Start(); j < C.End(); j++)
        {
            for (int l = C[j].Start(); l < C[j].End(); l++)
            {
                auto I = j - C.Start();
                auto J = l - C[j].Start();
                for (int n = 0; n < rows; n++)
                    C[j][l]+=bufferA.get()[I * rows + k] * bufferB.get()[J+rows *k];
            }
        }
        RotationVerticale(torus,bufferB,rows*cols);
    }

}

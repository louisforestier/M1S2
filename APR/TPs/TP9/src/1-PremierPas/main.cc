#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  
  // Get the number of processes
  int p;
  MPI_Comm_size(MPI_COMM_WORLD, &p); 
  
  // Get the rank of the process
  int q;
  MPI_Comm_rank(MPI_COMM_WORLD, &q);
  
  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  
  // Print a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n",
	 processor_name, q, p);
  
  // Finalize the MPI environment.
  MPI_Finalize();
  
  // bye
  return EXIT_SUCCESS;
}

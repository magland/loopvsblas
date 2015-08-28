#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIM (1024)
#define REPEATS ((int)(1e8)/DIM) //so we have about the same amount of run time for each choice of DIM
double A[DIM*REPEATS];
double B[DIM*REPEATS];

void main()
{
  double d;

  int i,j;

  srand(time(NULL));

  int Repeats=REPEATS; //so we don't risk doing the division at each iteration

  //Prepare all the random data ahead of time, so compiler can't be smart and avoid repeating calculations
  for (i=0; i<DIM*Repeats; ++i) {
    A[i]=rand()*1.0/rand();
    B[i]=rand()*1.0/rand();
  }

  printf("\nPerforming %d inner products of size %d\n\n",Repeats,DIM);

  printf("First we repeat the same random inner product many times:\n");
  {
    clock_t T1=clock();
    d = 0.0;
    for (j = 0; j < Repeats; ++j) {
      int offset=DIM*0;
      for (i = 0; i < DIM; ++i) {
        d += A[offset+i] * B[offset+i];
      }
    }
    printf("Result (loop) = %.0f\n",d);
    clock_t T2=clock();
    
    clock_t T3=clock();
    d=0.0;
    for (j = 0; j < Repeats; ++j) {
      d += cblas_ddot(DIM, &A[DIM*0], 1, &B[DIM*0], 1);
    }
    printf("Result (blas) = %.0f\n",d);
    clock_t T4=clock();

    double diff1=(T2-T1)*1.0;
    double diff2=(T4-T3)*1.0;
    printf("loop: %e blas: %e;  loop/cblas: %g\n",diff1,diff2,diff1/diff2);
  }
  printf("\n");

  printf("Next we use all different random inner products, so compiler can't be smart:\n");
  {
    clock_t T1=clock();
    d = 0.0;
    for (j = 0; j < Repeats; ++j) {
      int offset=DIM*j;
      for (i = 0; i < DIM; ++i) {
        d += A[offset+i] * B[offset+i];
      }
    }
    printf("Result (loop) = %.0f\n",d);
    clock_t T2=clock();
    
    clock_t T3=clock();
    d=0.0;
    for (j = 0; j < Repeats; ++j) {
      d += cblas_ddot(DIM, &A[DIM*j], 1, &B[DIM*j], 1);
    }
    printf("Result (blas) = %.0f\n",d);
    clock_t T4=clock();

    double diff1=(T2-T1)*1.0;
    double diff2=(T4-T3)*1.0;
    printf("loop: %e cblas: %e;  loop/cblas: %g\n",diff1,diff2,diff1/diff2);
  }
  printf("\n");

  printf("Finally we use a while loop instead of a for loop to avoid extra index arithmetic:\n");
  {
    clock_t T1=clock();
    d = 0.0;
    int cc=0;
    while (cc<DIM*Repeats) {
        d += A[cc] * B[cc];
        cc++;
    }
    printf("Result (loop) = %.0f\n",d);
    clock_t T2=clock();
    
    clock_t T3=clock();
    d=0.0;
    for (j = 0; j < Repeats; ++j) {
      d += cblas_ddot(DIM, &A[DIM*j], 1, &B[DIM*j], 1);
    }
    printf("Result (blas) = %.0f\n",d);
    clock_t T4=clock();

    double diff1=(T2-T1)*1.0;
    double diff2=(T4-T3)*1.0;
    printf("loop: %e cblas: %e;  loop/cblas: %g\n",diff1,diff2,diff1/diff2);
  }
  printf("\n");
  
}

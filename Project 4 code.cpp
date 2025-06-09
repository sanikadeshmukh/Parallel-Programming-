#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>
#include <xmmintrin.h>
#include <emmintrin.h>  // for __m128 and _mm_* intrinsics

#define SSE_WIDTH       4
#define ALIGNED         __attribute__((aligned(16)))
#define NUMTRIES        100

#ifndef ARRAYSIZE
#define ARRAYSIZE       1024*1024
#endif

ALIGNED float A[ARRAYSIZE];
ALIGNED float B[ARRAYSIZE];
ALIGNED float C[ARRAYSIZE];

void SimdMul(float*, float*, float*, int);
void SimdMulOMP(float*, float*, float*, int);  // Extra credit
void NonSimdMul(float*, float*, float*, int);
float SimdMulSum(float*, float*, int);
float NonSimdMulSum(float*, float*, int);

int main(int argc, char* argv[])
{
    omp_set_num_threads(atoi(argv[1]));

    for (int i = 0; i < ARRAYSIZE; i++) {
        A[i] = sqrtf((float)(i + 1));
        B[i] = sqrtf((float)(i + 1));
    }

    fprintf(stderr, "%12d\t", ARRAYSIZE);

    double maxPerformance = 0.;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        NonSimdMul(A, B, C, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance) maxPerformance = perf;
    }
    double megaMults = maxPerformance / 1000000.;
    fprintf(stderr, "N %10.2lf\t", megaMults);
    double mmn = megaMults;

    maxPerformance = 0.;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        SimdMul(A, B, C, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance) maxPerformance = perf;
    }
    megaMults = maxPerformance / 1000000.;
    fprintf(stderr, "S %10.2lf\t", megaMults);
    double mms = megaMults;
    double speedup = mms / mmn;
    fprintf(stderr, "(%6.2lf)\t", speedup);

    maxPerformance = 0.;
    float sumn, sums;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        sumn = NonSimdMulSum(A, B, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance) maxPerformance = perf;
    }
    double megaMultAdds = maxPerformance / 1000000.;
    fprintf(stderr, "N %10.2lf\t", megaMultAdds);
    mmn = megaMultAdds;

    maxPerformance = 0.;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        sums = SimdMulSum(A, B, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance) maxPerformance = perf;
    }
    megaMultAdds = maxPerformance / 1000000.;
    fprintf(stderr, "S %10.2lf\t", megaMultAdds);
    mms = megaMultAdds;
    speedup = mms / mmn;
    fprintf(stderr, "(%6.2lf)\n", speedup);

    // Extra Credit: SIMD + OpenMP
    maxPerformance = 0.;
    for (int t = 0; t < NUMTRIES; t++) {
        double time0 = omp_get_wtime();
        SimdMulOMP(A, B, C, ARRAYSIZE);
        double time1 = omp_get_wtime();
        double perf = (double)ARRAYSIZE / (time1 - time0);
        if (perf > maxPerformance) maxPerformance = perf;
    }
    megaMults = maxPerformance / 1000000.;
    fprintf(stderr, "SIMD+OpenMP %10.2lf\n", megaMults);

    return 0;
}

void NonSimdMul(float* A, float* B, float* C, int n)
{
    for (int i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

float NonSimdMulSum(float* A, float* B, int n)
{
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

void SimdMul(float* a, float* b, float* c, int len)
{
    int limit = (len / SSE_WIDTH) * SSE_WIDTH;
    __asm
    (
        ".att_syntax\n\t"
        "movq    -24(%rbp), %r8\n\t"
        "movq    -32(%rbp), %rcx\n\t"
        "movq    -40(%rbp), %rdx\n\t"
        );
    for (int i = 0; i < limit; i += SSE_WIDTH) {
        __asm
        (
            ".att_syntax\n\t"
            "movups     (%r8), %xmm0\n\t"
            "movups     (%rcx), %xmm1\n\t"
            "mulps      %xmm1, %xmm0\n\t"
            "movups     %xmm0, (%rdx)\n\t"
            "addq $16, %r8\n\t"
            "addq $16, %rcx\n\t"
            "addq $16, %rdx\n\t"
            );
    }
    for (int i = limit; i < len; i++) {
        c[i] = a[i] * b[i];
    }
}

float SimdMulSum(float* a, float* b, int len)
{
    float sum[4] = { 0., 0., 0., 0. };
    int limit = (len / SSE_WIDTH) * SSE_WIDTH;
    __asm
    (
        ".att_syntax\n\t"
        "movq    -40(%rbp), %r8\n\t"
        "movq    -48(%rbp), %rcx\n\t"
        "leaq    -32(%rbp), %rdx\n\t"
        "movups  (%rdx), %xmm2\n\t"
        );
    for (int i = 0; i < limit; i += SSE_WIDTH) {
        __asm
        (
            ".att_syntax\n\t"
            "movups     (%r8), %xmm0\n\t"
            "movups     (%rcx), %xmm1\n\t"
            "mulps      %xmm1, %xmm0\n\t"
            "addps      %xmm0, %xmm2\n\t"
            "addq $16, %r8\n\t"
            "addq $16, %rcx\n\t"
            );
    }
    __asm
    (
        ".att_syntax\n\t"
        "movups  %xmm2, (%rdx)\n\t"
        );
    for (int i = limit; i < len; i++) {
        sum[0] += a[i] * b[i];
    }
    return sum[0] + sum[1] + sum[2] + sum[3];
}

// ✅ EXTRA CREDIT: SIMD + OpenMP
void SimdMulOMP(float* a, float* b, float* c, int len)
{
    int limit = (len / SSE_WIDTH) * SSE_WIDTH;

#pragma omp parallel for
    for (int i = 0; i < limit; i += SSE_WIDTH)
    {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vc = _mm_mul_ps(va, vb);
        _mm_storeu_ps(&c[i], vc);
    }

    for (int i = limit; i < len; i++)
    {
        c[i] = a[i] * b[i];
    }
}



#!/bin/bash

# Output file
OUTFILE = "full_results.csv"
echo "ArraySize,NonSIMD_Mults/sec,SIMD_Mults/sec,Speedup_Mul,NonSIMD_Mults+Adds/sec,SIMD_Mults+Adds/sec,Speedup_Sum" > $OUTFILE

# List of array sizes to test
for size in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
do
echo "Testing size $size..."

# Modify ARRAYSIZE in the source code
sed "s/^#define ARRAYSIZE.*/#define ARRAYSIZE $size/" proje4.cpp > project4


g++ - fopenmp proje4.cpp - o project4

# Run and capture output
OUTPUT = $(. / p4)

# Extract values
Speedup_Mul = $(echo "$OUTPUT" | grep "Max Speedup (C" | awk '{print $NF}')
Speedup_Sum = $(echo "$OUTPUT" | grep "Max Speedup (sum" | awk '{print $NF}')

Perf_NonSIMD_Mul = $(echo "$OUTPUT" | grep "NonSIMD Performance = .*MegaMultipliesPerSecond" | awk - F'=' '{print $2}' | awk '{print $1}')
Perf_SIMD_Mul = $(echo "$OUTPUT" | grep "SIMD Performance = .*MegaMultipliesPerSecond" | awk - F'=' '{print $2}' | awk '{print $1}')
Perf_NonSIMD_Sum = $(echo "$OUTPUT" | grep "NonSIMD Performance = .*MegaMultipliesAndAddsPerSecond" | awk - F'=' '{print $2}' | awk '{print $1}')
Perf_SIMD_Sum = $(echo "$OUTPUT" | grep "SIMD Performance = .*MegaMultipliesAndAddsPerSecond" | awk - F'=' '{print $2}' | awk '{print $1}')
# Append to CSV
echo "$size,$Perf_NonSIMD_Mul,$Perf_SIMD_Mul,$Speedup_Mul,$Perf_NonSIMD_Sum,$Perf_SIMD_Sum,$Speedup_Sum" >> $OUTFILE
done

echo "Done. Results saved to $OUTFILE"
# -------------------------------
# Extra Credit : SIMD + OpenMP Test
# -------------------------------

echo "NumThreads,SIMD+OpenMP_Perf" > extra_credit_results.csv

for threads in 1 2 4 8
do
echo "Running with $threads thread(s)..."
# Set thread count as argument to project4
output = $(. / project4 $threads 2 > &1)
# Extract SIMD + OpenMP performance value from output
perf = $(echo "$output" | grep "SIMD+OpenMP" | awk '{print $2}')

echo "$threads,$perf" >> extra_credit_results.csv
done

echo "Extra credit results saved to extra_credit_results.csv"

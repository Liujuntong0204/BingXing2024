#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h> 
#include <immintrin.h> //AVX、AVX2
using namespace std;
int N = 1000;
#define NUM_THREADS 4
float** A = NULL;  //系数数组

//long long head, tail, freq;

//定义计时变量
struct timespec start;
struct timespec end1;
time_t total_duration_sec;
long total_duration_nsec;


void A_init() {     //数组初始化
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand() % 1000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 1000;
            }
        }
    }
}


void deleteA() { //数组重置
    for (int i = 0; i < N; i++) {
        delete[] A[i];
    }
    delete A;
}


//串行算法
void LU() {   
    int i, j, k;
    for (k = 0; k < N; k++) {
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

//串行算法＋openmp_static
void LU_omp() {   
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        A[k][k] = 1.0;
        }

#pragma omp for
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

//串行算法＋openmp+guided
void LU_omp_guided() {    
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

//串行算法＋openmp+dynamic
void LU_omp_dynamic() {    
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}


//avx优化算法
void avx_optimized() {
    for (int k = 0; k < N; k++) {
        __m256 t1 = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8) {
            __m256 t2 = _mm256_loadu_ps(&A[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&A[k][j], t2);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}



//avx+openmp+static
void avx_omp_static() {   
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]); 
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(static)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

//avx+openmp+dynamic
void avx_omp_dynamic() {            
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);  
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

//avx+openmp＋guided
void avx_omp_guided() {            
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);   
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void cal(void(*func)()) {
    A_init();
    timespec_get(&start, TIME_UTC);
    func();
    timespec_get(&end1, TIME_UTC);

}


int main() {
    for(int i=0;i<6;i++)
    {
        cin >> N;
        cal(LU);
        total_duration_sec = end1.tv_sec - start.tv_sec;
        total_duration_nsec = end1.tv_nsec - start.tv_nsec;
        if (total_duration_nsec < 0)
        {
            total_duration_sec--;
            total_duration_nsec += 1000000000L;
        }
        printf("Time of serial : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
        deleteA();

        cal(LU_omp);
        total_duration_sec = end1.tv_sec - start.tv_sec;
        total_duration_nsec = end1.tv_nsec - start.tv_nsec;
        if (total_duration_nsec < 0)
        {
            total_duration_sec--;
            total_duration_nsec += 1000000000L;
        }
        printf("Time of NoSimd+openmp_static : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
        deleteA();


        cal(LU_omp_dynamic);
        total_duration_sec = end1.tv_sec - start.tv_sec;
        total_duration_nsec = end1.tv_nsec - start.tv_nsec;
        if (total_duration_nsec < 0)
        {
            total_duration_sec--;
            total_duration_nsec += 1000000000L;
        }
        printf("Time of NoSimd+openmp_dynamic : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
        deleteA();

        cal(LU_omp_guided);
        total_duration_sec = end1.tv_sec - start.tv_sec;
        total_duration_nsec = end1.tv_nsec - start.tv_nsec;
        if (total_duration_nsec < 0)
        {
            total_duration_sec--;
            total_duration_nsec += 1000000000L;
        }
        printf("Time of NoSimd+openmp_guided : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
        deleteA();

        cal(avx_optimized);
        total_duration_sec = end1.tv_sec - start.tv_sec;
        total_duration_nsec = end1.tv_nsec - start.tv_nsec;
        if (total_duration_nsec < 0)
        {
            total_duration_sec--;
            total_duration_nsec += 1000000000L;
        }
        printf("Time of SIMD : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
        deleteA();

        cal(avx_omp_static);
        total_duration_sec = end1.tv_sec - start.tv_sec;
        total_duration_nsec = end1.tv_nsec - start.tv_nsec;
        if (total_duration_nsec < 0)
        {
            total_duration_sec--;
            total_duration_nsec += 1000000000L;
        }
        printf("Time of SIMD+openmp_static : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
        deleteA();

        cal(avx_omp_dynamic);
        total_duration_sec = end1.tv_sec - start.tv_sec;
        total_duration_nsec = end1.tv_nsec - start.tv_nsec;
        if (total_duration_nsec < 0)
        {
            total_duration_sec--;
            total_duration_nsec += 1000000000L;
        }
        printf("Time of SIMD+openmp_dynamic : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
        deleteA();

        cal(avx_omp_guided);
        total_duration_sec = end1.tv_sec - start.tv_sec;
        total_duration_nsec = end1.tv_nsec - start.tv_nsec;
        if (total_duration_nsec < 0)
        {
            total_duration_sec--;
            total_duration_nsec += 1000000000L;
        }
        printf("Time of SIMD+openmp_guided : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
        deleteA();
    }
}
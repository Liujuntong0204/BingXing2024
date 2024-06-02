#include<iostream>
#include <stdio.h>
#include<cstring>
#include<typeinfo>
#include <stdlib.h>
#include<cmath>
#include<mpi.h>
#include<omp.h>

#include<immintrin.h>
using namespace std;
int N = 500;
#define NUM_THREADS 7
float** A = NULL;

void A_init() {     //δ���������ĳ�ʼ��
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand() % 5000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 5000;
            }
        }
    }
}



void deleteA() {
    for (int i = 0; i < N; i++) {
        delete[] A[i];
    }
    delete A;
}

void ReGetValue()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = 0;
        }
    }
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand() % 5000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 5000;
            }
        }
    }
}


//�黮��MPI ������ͨ�� openmp
double LU_mpi_async_omp(int argc, char* argv[]) {  //������ͨ��+OpenMP
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    cout << MPI_Wtick();
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {  //0�Ž��̳�ʼ������
        A_init();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//���������ݾ�������
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //�ȴ�����

    }
    else {
        A_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            if ((begin <= k && k < end)) {
                for (j = k + 1; j < N; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];  //����������
                for (j = 0; j < total; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < N / total * total)//�ڿɾ��ֵ���������
                    src = k / (N / total);
                else
                    src = total - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
            }
        }
#pragma omp for schedule(guided)  //��ʼ���߳�
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//������ͬ��
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI���黮��+������+OpenMP��ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}

//�黮��MPI ������ͨ�� openmp avx
double LU_mpi_async_multithread(int argc, char* argv[]) {  //������ͨ��+avx+omp
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    cout << MPI_Wtick();
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {  //0�Ž��̳�ʼ������
        A_init();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//���������ݾ�������
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //�ȴ�����

    }
    else {
        A_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            if ((begin <= k && k < end)) {
                __m256 t1 = _mm256_set1_ps(A[k][k]);
                for (j = k + 1; j + 8 <= N; j += 8) {
                    __m256 t2 = _mm256_loadu_ps(&A[k][j]);  //AVX�Ż���������
                    t2 = _mm256_div_ps(t2, t1);
                    _mm256_storeu_ps(&A[k][j], t2);
                }
                for (; j < N; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];  //����������
                for (j = 0; j < total; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < N / total * total)//�ڿɾ��ֵ���������
                    src = k / (N / total);
                else
                    src = total - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
            }
        }
#pragma omp for schedule(guided)  //��ʼ���߳�
        for (i = max(begin, k + 1); i < end; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);   //AVX�Ż���ȥ����
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
    MPI_Barrier(MPI_COMM_WORLD);	//������ͬ��
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI���黮��+������+OpenMP+AVX��ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}

//�黮��MPI ������ͨ�� avx
double LU_mpi_async_avx(int argc, char* argv[]) {  //������ͨ��+avx
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    cout << MPI_Wtick();
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {  //0�Ž��̳�ʼ������
        A_init();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//���������ݾ�������
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //�ȴ�����

    }
    else {
        A_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //����������
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        {
            if ((begin <= k && k < end)) {
                __m256 t1 = _mm256_set1_ps(A[k][k]);
                for (j = k + 1; j + 8 <= N; j += 8) {
                    __m256 t2 = _mm256_loadu_ps(&A[k][j]);  //AVX�Ż���������
                    t2 = _mm256_div_ps(t2, t1);
                    _mm256_storeu_ps(&A[k][j], t2);
                }
                for (; j < N; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];  //����������
                for (j = 0; j < total; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < N / total * total)//�ڿɾ��ֵ���������
                    src = k / (N / total);
                else
                    src = total - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
            }
        }
        for (i = max(begin, k + 1); i < end; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);   //AVX�Ż���ȥ����
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
    MPI_Barrier(MPI_COMM_WORLD);	//������ͬ��
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI���黮��+������+AVX��ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}

int main(int argc, char* argv[]) {
    /*   QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
       cal(LU);
       cout << "ƽ���㷨��ʱ��" << (tail - head) * 1000 / freq << "ms" << endl;
       deleteA();*/

    for (int i = 1; i <= 6; i++)
    {
        N = 500 * i;
        A_init();//����
        //�黮��MPI������openmp
        LU_mpi_async_omp(argc, argv);
        ReGetValue();//��ֵ
        //�黮��MPI������openmp+avx
        LU_mpi_async_multithread(argc, argv);
        ReGetValue();//��ֵ
        //���ظ����ȵĿ黮��MPI����ͨ��
        LU_mpi_async_avx(argc, argv);
        deleteA();

    }
}



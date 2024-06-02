#include<iostream>
#include <stdio.h>
#include<cstring>
#include<typeinfo>
#include <stdlib.h>
#include<cmath>
#include<mpi.h>

using namespace std;
int N = 500;
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

void A_initAsEmpty() {
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
        memset(A[i], 0, N * sizeof(float));
    }

}


//�黮�ֵ�MPI  ����ͨ��
double LU_mpi(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
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

        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1�ǳ�ʼ������Ϣ����ÿ�����̷�������
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);  //��ʱÿ�����̶��õ�������
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < total; j++) { //
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0����Ϣ��ʾ�������
            }
        }
        else {
            int src;
            if (k < N / total * total)//�ڿɾ��ֵ���������
                src = k / (N / total);
            else
                src = total - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//������ͬ��
    if (rank == 0) {//0�Ž����д������ս��
        end_time = MPI_Wtime();
        printf("�黮��MPI����ͨ�ź�ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

//�黮��MPI ������ͨ��
double LU_mpi_async(int argc, char* argv[]) {  
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
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
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];  //����������
            for (j = rank + 1; j < total; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //��ִ������������񣬿�ֱ������
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
        }
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
        printf("�黮��MPI������ͨ�ź�ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}


//ѭ������MPI ����ͨ��
double LU_mpi_circle(int argc, char* argv[]) { 
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {  //0�Ž��̳�ʼ������
        A_init();

        for (j = 1; j < total; j++) {
            for (i = j; i < N; i += total) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1�ǳ�ʼ������Ϣ����ÿ�����̷�������
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = rank; i < N; i += total) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if (k % total == rank) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < total; j++) { //
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0����Ϣ��ʾ�������
            }
        }
        else {
            int src = k % total;

            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        int begin = k;
        while (begin % total != rank)
            begin++;
        for (i = begin; i < N; i += total) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//������ͬ��
    if (rank == 0) {//0�Ž����д������ս��
        end_time = MPI_Wtime();
        printf("ѭ������MPI����ͨ�ź�ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);

    }
    MPI_Finalize();
    return end_time - start_time;
}

//���ظ��Ӿ��ȵĿ黮��MPI ����ͨ��
double LU_mpi_withExtra(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    int extra = -1;
    bool ifExtraDone = true;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = N / total * (rank + 1);
    if (rank < N % total) {
        extra = N / total * total + rank;
        ifExtraDone = false;
    }
    if (rank == 0) {  //0�Ž��̳�ʼ������
        A_init();
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1�ǳ�ʼ������Ϣ����ÿ�����̷�������
            }
        }
        if (extra != -1) {
            for (i = 1; i < N % total; i++) {
                MPI_Send(&A[N / total * total + i][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD);  //β���Ķ������񣬷��͸���Ӧ����Ľ���
            }
        }
    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
        if (extra != -1) {
            MPI_Recv(&A[extra][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end) || (k == extra)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < total; j++) {
                if (j != rank) {
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0����Ϣ��ʾ�������
                }
            }
        }
        else {
            int src;
            if (k < N / total * total)//�ڿɾ��ֵ���������
                src = k / (N / total);
            else
                src = k - (N / total * total);

            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);

        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        if (!ifExtraDone) {           //���ж��⸺�صĽ��̣�ÿ��Ҫ�����һ�е���ȥ
            for (j = k + 1; j < N; j++) {
                A[extra][j] = A[extra][j] - A[extra][k] * A[k][j];
            }
            A[extra][k] = 0;
            if (extra == k + 1) {
                ifExtraDone = true;
            }
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);	//������ͬ��
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("���ظ�����Ŀ黮��MPI����ͨ�ź�ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

//��ˮ���㷨
double LU_mpi_pipeline(int argc, char* argv[]) {  
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
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

        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1�ǳ�ʼ������Ϣ����ÿ�����̷�������
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];  //����������
            for (j = rank + 1; j < total; j++) { //�黮���У��Ѿ���Ԫ���ҽ����˳�����1����������

                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0����Ϣ��ʾ�������
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //��ִ������������񣬿�ֱ������
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //ʵ������Ȼ���������գ���Ϊ�������Ĳ�����Ҫ��Щ����
        }
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
        printf("�黮��MPI ��ˮ���㷨��%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}




//void cal(void(*func)()) {
//    A_init();
//    func();
//}

int main(int argc, char* argv[]) {
    
    //for (int i = 1; i <= 6; i++)
    //{
    //    N = 500 * i;
    //    A_init();//����
    //    //�黮��MPI����ͨ��
    //    LU_mpi(argc, argv);
    //    ReGetValue();//��ֵ
    //    //���ظ����ȵĿ黮��MPI����ͨ��
    //    LU_mpi_withExtra(argc, argv);
    //    ReGetValue();//��ֵ
    //    //ѭ������MPI����ͨ��
    //    LU_mpi_circle(argc, argv);
    //    ReGetValue();//��ֵ
    //    //�黮��MPI������ͨ��
    //    LU_mpi_async(argc, argv);
    //    ReGetValue();//��ֵ
    //    //�黮��MPI ��ˮ���㷨
    //    LU_mpi_pipeline(argc, argv);
    //    deleteA();

    //}
    N = 500;
    cout << "N=" << N << endl;
    A_init();//����
    //�黮��MPI����ͨ��
    LU_mpi(argc, argv);
}


#include<iostream>
#include <stdio.h>
#include<cstring>
#include<typeinfo>
#include <stdlib.h>
#include<cmath>
#include <time.h> 

using namespace std;
int N = 500;
float** A = NULL;

struct timespec start;
struct timespec end1;
time_t total_duration_sec;
long total_duration_nsec;

void A_init() {     //未对齐的数组的初始化
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

//普通串行算法
void LU() {
    timespec_get(&start, TIME_UTC);
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    timespec_get(&end1, TIME_UTC);
    total_duration_sec = end1.tv_sec - start.tv_sec;
    total_duration_nsec = end1.tv_nsec - start.tv_nsec;
    if (total_duration_nsec < 0)
    {
        total_duration_sec--;
        total_duration_nsec += 1000000000L;
    }
    printf("Time of serial: %lld.%09lds\n", total_duration_sec, total_duration_nsec);
}
int main()
{
    for (int i = 1; i <= 6; i++)
    {
        N = 500 * i;
        //串行算法
        A_init();//声明
        LU();
        deleteA();//赋0
    }
}
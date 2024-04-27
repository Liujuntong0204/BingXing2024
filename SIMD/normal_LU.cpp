#include<iostream>
#include <stdio.h>  
#include <time.h> 
using namespace std;
int main()
{
	int N = 200;//数据规模
	for (int x = 0; x < 20; x++)
	{
		cout << "Data scale:" << N << endl;
		float* anwser = new float[N];//结果数组
		//二维数组初始化
		float** arry = new float* [N];
		for (int i = 0; i < N; i++)
		{
			arry[i] = new float[N];
		}
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				arry[i][j] = rand() % 1000 + 1;//避免有0
			}
		}
		//常数列数组初始化
		float* con = new float[N];
		for (int i = 0; i < N; i++)
		{
			con[i] = rand() % 1000;
		}
		//计时
		struct timespec start1, end1, start2, end2;
		time_t duration_sec1, duration_sec2;
		long duration_nsec1, duration_nsec2;
		time_t total_duration_sec;
		long total_duration_nsec;
		//平凡算法----------------------------------------------------------------------------------------
		//消元过程的时间测量  
		timespec_get(&start1, TIME_UTC);
		//消元
		for (int k = 0; k < N; k++)
		{
			for (int i = k + 1; i < N; i++)
			{
				//求出相差倍数
				float temp = arry[i][k] / arry[k][k];
				//遍历这一行的所有值，将从i开始的数值依次减去相对应的值乘以倍数
				for (int j = k; j < N; j++)
				{
					arry[i][j] -= arry[k][j] * temp;
				}
				con[i] -= temp * con[k];
			}
		}
		timespec_get(&end1, TIME_UTC);
		duration_sec1 = end1.tv_sec - start1.tv_sec;
		duration_nsec1 = end1.tv_nsec - start1.tv_nsec;
		if (duration_nsec1 < 0)
		{
			duration_sec1--;
			duration_nsec1 += 1000000000L;
		}
		printf("Time of elimination: %lld.%09lds\n", duration_sec1, duration_nsec1);

		// 回代过程的时间测量  
		timespec_get(&start2, TIME_UTC);
		//回代
		//从最后一个未知数开始求，依次向上求解
		anwser[N - 1] = con[N - 1] / arry[N - 1][N - 1];
		for (int i = N - 2; i >= 0; i--)
		{
			float sum = con[i];
			for (int j = i + 1; j < N; j++)
			{
				sum -= arry[i][j] * anwser[j];
			}
			anwser[i] = sum / arry[i][i];
		}
		timespec_get(&end2, TIME_UTC);
		duration_sec2 = end2.tv_sec - start2.tv_sec;
		duration_nsec2 = end2.tv_nsec - start2.tv_nsec;
		if (duration_nsec2 < 0) {
			duration_sec2--;
			duration_nsec2 += 1000000000L;
		}
		printf("Time of execution: %lld.%09lds\n", duration_sec2, duration_nsec2);
		//for (int i = 0; i < N; i++)
		//	cout << anwser[i] << " ";
		//cout << endl;

		// 计算总时间  
		total_duration_sec = duration_sec1 + duration_sec2;
		total_duration_nsec = duration_nsec1 + duration_nsec2;
		if (total_duration_nsec >= 1000000000L) {
			total_duration_sec++;
			total_duration_nsec -= 1000000000L;
		}
		printf("Time of all: %lld.%09lds\n", total_duration_sec, total_duration_nsec);

		N += 200;
	}

}
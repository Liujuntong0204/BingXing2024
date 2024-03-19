#include<iostream>
#include<windows.h>
using namespace std;
//第一题 计算n*n矩阵的每一列与给定向量的内积
int main()
{
	for(int p=0;p<14;p++)
	{
		int n;
		cin >> n;
		int** arry = new int* [n];
		for (int i = 0; i < n; i++)
			arry[i] = new int[n];
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				arry[i][j] = i + j;
		//cin >> arry[i][j];//得到二维数组arry[n][n]
//给定向量
		int* vector = new int[n];
		for (int i = 0; i < n; i++)
			vector[i] = i;
		// cin >> vector[i];
		// 

	//方法1，逐列访问元素的平凡算法（按列访问二维数组）
		int* answer1 = new int[n];
		for (int i = 0; i < n; i++)
			answer1[i] = 0;

		//计时
		double time1 = 0;
		LARGE_INTEGER nFreq;
		LARGE_INTEGER nBeginTime;
		LARGE_INTEGER nEndTime;
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&nBeginTime);//开始计时

		//计算内积，储存在anwser数组中
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				answer1[i] += arry[j][i] * vector[j];
			}
		}


		////输出answer结果数组
		//for (int i = 0; i < n; i++)
		//	cout << answer1[i] << " ";
		//cout << endl;

		QueryPerformanceCounter(&nEndTime);//停止计时
		time1 = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
		cout << "方法1运行的时间是" << time1 * 1000 << "ms" << endl;

		//***************************************************************************

		//方法2，cache优化算法（按行访问二维数组）

		int* answer2 = new int[n];
		for (int i = 0; i < n; i++)
			answer2[i] = 0;

				//计时
		double time2 = 0;
		LARGE_INTEGER nFreq2;
		LARGE_INTEGER nBeginTime2;
		LARGE_INTEGER nEndTime2;
		QueryPerformanceFrequency(&nFreq2);
		QueryPerformanceCounter(&nBeginTime2);//开始计时




		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				answer2[j] += arry[i][j] * vector[i];
			}
		}

		//for (int i = 0; i < n; i++)
		//	cout << answer2[i] << " ";
		//cout << endl;

		QueryPerformanceCounter(&nEndTime2);//停止计时
		time2 = (double)(nEndTime2.QuadPart - nBeginTime2.QuadPart) / (double)nFreq2.QuadPart;
		cout << "方法2运行的时间是" << time2 * 1000 << "ms" << endl;

	}


}
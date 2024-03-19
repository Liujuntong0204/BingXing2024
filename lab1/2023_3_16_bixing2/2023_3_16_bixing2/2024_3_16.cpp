#include<iostream>
#include<windows.h>
using namespace std;
//第二题 计算n个数的和
int add(int* arry, int start, int end)//递归函数
{
	if (start == end)
		return arry[start];
	else
		return add(arry, start, start + (end - start) / 2) + add(arry, start + (end - start) / 2 + 1, end);

}
int main()
{
	for(int p=0;p<20;p++)
	{
		int n;
		cin >> n;
		int* arry = new int[n];
		for (int i = 0; i < n; i++)
			//cin >> arry[i];//给定数组
			arry[i] = i;
		int answer1 = 0;//储存答案值
		//方法1，逐个累加的平凡算法（链式

		//计时
		double time1 = 0;
		LARGE_INTEGER nFreq;
		LARGE_INTEGER nBeginTime;
		LARGE_INTEGER nEndTime;
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&nBeginTime);//开始计时



		for (int i = 0; i < n; i++)
			answer1 += arry[i];
		//cout << answer1 << endl;


		QueryPerformanceCounter(&nEndTime);//停止计时
		time1 = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
		cout << "方法1运行的时间是" << time1 * 1000 << "ms" << endl;

		//***************************************************************************************
		//方法2，超标量优化算法（递归

		//计时
		double time2 = 0;
		LARGE_INTEGER nFreq2;
		LARGE_INTEGER nBeginTime2;
		LARGE_INTEGER nEndTime2;
		QueryPerformanceFrequency(&nFreq2);
		QueryPerformanceCounter(&nBeginTime2);//开始计时

		int answer2 = add(arry, 0, n - 1);
		//cout << answer2 << endl;

		QueryPerformanceCounter(&nEndTime2);//停止计时
		time2 = (double)(nEndTime2.QuadPart - nBeginTime2.QuadPart) / (double)nFreq2.QuadPart;
		cout << "方法2运行的时间是" << time2 * 1000 << "ms" << endl;


		//方法三，双线程并行


		int sum1 = 0;
		int sum2 = 0;
		int answer3 = 0;
		double time3 = 0;
		LARGE_INTEGER nFreq3;
		LARGE_INTEGER nBeginTime3;
		LARGE_INTEGER nEndTime3;
		QueryPerformanceFrequency(&nFreq3);
		QueryPerformanceCounter(&nBeginTime3);//开始计时
		for (int i = 0; i < n; i += 2)
		{
			sum1 += arry[i];
			sum2 += arry[i + 1];
		}
		answer3 = sum1 + sum2;
		//cout << answer3 << endl;
		QueryPerformanceCounter(&nEndTime3);//停止计时
		time3 = (double)(nEndTime3.QuadPart - nBeginTime3.QuadPart) / (double)nFreq3.QuadPart;
		cout << "方法3运行的时间是" << time3 * 1000 << "ms" << endl;
	}


}
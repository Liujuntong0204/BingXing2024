#include<iostream>
#include<windows.h>
using namespace std;
//��һ�� ����n*n�����ÿһ��������������ڻ�
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
		//cin >> arry[i][j];//�õ���ά����arry[n][n]
//��������
		int* vector = new int[n];
		for (int i = 0; i < n; i++)
			vector[i] = i;
		// cin >> vector[i];
		// 

	//����1�����з���Ԫ�ص�ƽ���㷨�����з��ʶ�ά���飩
		int* answer1 = new int[n];
		for (int i = 0; i < n; i++)
			answer1[i] = 0;

		//��ʱ
		double time1 = 0;
		LARGE_INTEGER nFreq;
		LARGE_INTEGER nBeginTime;
		LARGE_INTEGER nEndTime;
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&nBeginTime);//��ʼ��ʱ

		//�����ڻ���������anwser������
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				answer1[i] += arry[j][i] * vector[j];
			}
		}


		////���answer�������
		//for (int i = 0; i < n; i++)
		//	cout << answer1[i] << " ";
		//cout << endl;

		QueryPerformanceCounter(&nEndTime);//ֹͣ��ʱ
		time1 = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
		cout << "����1���е�ʱ����" << time1 * 1000 << "ms" << endl;

		//***************************************************************************

		//����2��cache�Ż��㷨�����з��ʶ�ά���飩

		int* answer2 = new int[n];
		for (int i = 0; i < n; i++)
			answer2[i] = 0;

				//��ʱ
		double time2 = 0;
		LARGE_INTEGER nFreq2;
		LARGE_INTEGER nBeginTime2;
		LARGE_INTEGER nEndTime2;
		QueryPerformanceFrequency(&nFreq2);
		QueryPerformanceCounter(&nBeginTime2);//��ʼ��ʱ




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

		QueryPerformanceCounter(&nEndTime2);//ֹͣ��ʱ
		time2 = (double)(nEndTime2.QuadPart - nBeginTime2.QuadPart) / (double)nFreq2.QuadPart;
		cout << "����2���е�ʱ����" << time2 * 1000 << "ms" << endl;

	}


}
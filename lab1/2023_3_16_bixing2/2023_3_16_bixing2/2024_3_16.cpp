#include<iostream>
#include<windows.h>
using namespace std;
//�ڶ��� ����n�����ĺ�
int add(int* arry, int start, int end)//�ݹ麯��
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
			//cin >> arry[i];//��������
			arry[i] = i;
		int answer1 = 0;//�����ֵ
		//����1������ۼӵ�ƽ���㷨����ʽ

		//��ʱ
		double time1 = 0;
		LARGE_INTEGER nFreq;
		LARGE_INTEGER nBeginTime;
		LARGE_INTEGER nEndTime;
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&nBeginTime);//��ʼ��ʱ



		for (int i = 0; i < n; i++)
			answer1 += arry[i];
		//cout << answer1 << endl;


		QueryPerformanceCounter(&nEndTime);//ֹͣ��ʱ
		time1 = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
		cout << "����1���е�ʱ����" << time1 * 1000 << "ms" << endl;

		//***************************************************************************************
		//����2���������Ż��㷨���ݹ�

		//��ʱ
		double time2 = 0;
		LARGE_INTEGER nFreq2;
		LARGE_INTEGER nBeginTime2;
		LARGE_INTEGER nEndTime2;
		QueryPerformanceFrequency(&nFreq2);
		QueryPerformanceCounter(&nBeginTime2);//��ʼ��ʱ

		int answer2 = add(arry, 0, n - 1);
		//cout << answer2 << endl;

		QueryPerformanceCounter(&nEndTime2);//ֹͣ��ʱ
		time2 = (double)(nEndTime2.QuadPart - nBeginTime2.QuadPart) / (double)nFreq2.QuadPart;
		cout << "����2���е�ʱ����" << time2 * 1000 << "ms" << endl;


		//��������˫�̲߳���


		int sum1 = 0;
		int sum2 = 0;
		int answer3 = 0;
		double time3 = 0;
		LARGE_INTEGER nFreq3;
		LARGE_INTEGER nBeginTime3;
		LARGE_INTEGER nEndTime3;
		QueryPerformanceFrequency(&nFreq3);
		QueryPerformanceCounter(&nBeginTime3);//��ʼ��ʱ
		for (int i = 0; i < n; i += 2)
		{
			sum1 += arry[i];
			sum2 += arry[i + 1];
		}
		answer3 = sum1 + sum2;
		//cout << answer3 << endl;
		QueryPerformanceCounter(&nEndTime3);//ֹͣ��ʱ
		time3 = (double)(nEndTime3.QuadPart - nBeginTime3.QuadPart) / (double)nFreq3.QuadPart;
		cout << "����3���е�ʱ����" << time3 * 1000 << "ms" << endl;
	}


}
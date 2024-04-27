#include<iostream>
#include <stdio.h>  
#include <time.h> 
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2
using namespace std;
int main()
{
	const int N = 4000;//���ݹ�ģ
	for (int x = 0; x < 1; x++)
	{
		cout << "Data scale:" << N << endl;
		//float* anwser = (float*)_aligned_malloc(N * sizeof(float), 16);//�������
		//��ά�����ʼ��������
		//float** arry=(float**)_aligned_malloc(N * sizeof(float*), 16);
		//for (int i = 0; i < N; i++)
		//{
		//	arry[i]= (float*)_aligned_malloc(N * sizeof(float), 16);
		//}
		float* arry1 = (float*)_aligned_malloc(N * N * sizeof(float), 16);
		float* con = (float*)_aligned_malloc(N * sizeof(float), 16);
		float* anwser= (float*)_aligned_malloc(N * sizeof(float), 16);
		float(*arry)[N] = reinterpret_cast<float(*)[(N)]>(arry1);
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				arry[i][j] = rand() % 1000 + 1;
			}
		}
		//�����������ʼ��
		/*float* con = (float*)_aligned_malloc(N * sizeof(float), 16);*/
		for (int i = 0; i < N; i++)
		{
			con[i] = rand() % 1000;
		}
		//��ʱ
		struct timespec start1, end1, start2, end2;
		time_t duration_sec1, duration_sec2;
		long duration_nsec1, duration_nsec2;
		time_t total_duration_sec;
		long total_duration_nsec;
		//ƽ���㷨----------------------------------------------------------------------------------------
		//��Ԫ���̵�ʱ�����  
		timespec_get(&start1, TIME_UTC);
		//��Ԫ
		for (int k = 0; k < N; k++)
		{
			//���õ�k�е���Ԫarry[k][k]��Ϊ1
			__m128 vt = _mm_set1_ps(arry[k][k]);//��ά����
			int j = 0;
			for (j = k + 1; j + 3 < N; j += 4)
			{
				__m128 va = _mm_load_ps(&arry[k][j]);
				__m128 va_1 = _mm_div_ps(va, vt);
				_mm_store_ps(&arry[k][j], va_1);
			}
			for (; j < N; j++)//����ʣ��Ԫ��
			{
				arry[k][j] = arry[k][j] / arry[k][k];
			}
			arry[k][k] = 1.0f;
			int i = 0;
			for (i = k + 1; i < N; i++)
			{
				__m128 vaik = _mm_set1_ps(arry[i][k]);//������Ԫ�أ���Ӧ���ű���
				int j = 0;
				for (j = k + 1; j + 3 < N; j += 4)
				{
					__m128 vakj = _mm_load_ps(&arry[k][j]);
					__m128 vaij = _mm_load_ps(&arry[i][j]);
					__m128 vx = _mm_mul_ps(vakj, vaik);
					__m128 vaij_1 = _mm_sub_ps(vaij, vx);
					_mm_store_ps(&arry[i][j], vaij_1);
				}
				for (; j < N; j++)
				{
					arry[i][j] = arry[i][j] - arry[k][j] * arry[i][k];
				}
				arry[i][k] = 0;
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

		// �ش����̵�ʱ�����  
		timespec_get(&start2, TIME_UTC);
		//�ش�
		//�����һ��δ֪����ʼ�������������
		anwser[N - 1] = con[N - 1] / arry[N - 1][N - 1];
		__m128 sumve;
		for (int i = N - 2; i >= 0; i--)
		{
			int j;
			for (j = i + 1; j + 3 < N; j += 4)
			{
				__m128 vaij = _mm_load_ps(&arry[i][j]);//���ظ��к���4��ֵ
				__m128 vaan = _mm_load_ps(&anwser[j]);//���ض�Ӧ�Ĵ�ֵ
				__m128 vaij_1 = _mm_mul_ps(vaij, vaan);
				float sum[4] = { 0 };
				//_mm_store_ps(&arry[i][j], vaij_1);
				_mm_store_ps(sum, vaij_1);
				float total = sum[0] + sum[1] + sum[2] + sum[3];
				con[i] = con[i] - total;
			}
			for (; j < N; j++)//����ʣ��Ԫ��
			{
				con[i] -= arry[i][j] * anwser[j];
			}
			anwser[i] = con[i] / arry[i][i];
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

		// ������ʱ��  
		total_duration_sec = duration_sec1 + duration_sec2;
		total_duration_nsec = duration_nsec1 + duration_nsec2;
		if (total_duration_nsec >= 1000000000L) {
			total_duration_sec++;
			total_duration_nsec -= 1000000000L;
		}
		printf("Time of all: %lld.%09lds\n", total_duration_sec, total_duration_nsec);

		//N += 200;
	}

}
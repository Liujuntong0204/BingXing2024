#include<iostream>
#include <stdio.h>  
#include <time.h> 
#include<arm_neon.h>
#include<pthread.h>
#include <stdlib.h>
#include<semaphore.h>
using namespace std;

int n = 1000;
float** A = NULL;
const int NUM_THREADS = 7;

//�����ʱ����
struct timespec start;
struct timespec end1;
time_t total_duration_sec;
long total_duration_nsec;




void A_init() {     //δ���������ĳ�ʼ��
	A = new float* [n];
	for (int i = 0; i < n; i++) {
		A[i] = new float[n];
	}
	for (int i = 0; i < n; i++) {
		A[i][i] = 1.0;
		for (int j = i + 1; j < n; j++) {
			A[i][j] = rand() % 1000;
		}

	}
	for (int k = 0; k < n; k++) {
		for (int i = k + 1; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A[i][j] += A[k][j];
				A[i][j] = (int)A[i][j] % 1000;
			}
		}
	}
}

void deleteA() {
	for (int i = 0; i < n; i++) {
		delete[] A[i];
	}
	delete A;
}

void f_ordinary()
{
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];
		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}


struct threadParam_t
{
	int t_id; //�߳� id
};

//�ź�������
sem_t sem_leader;
sem_t* sem_Divsion = new sem_t[NUM_THREADS - 1]; // ÿ���߳����Լ�ר�����ź���
sem_t* sem_Elimination = new sem_t[NUM_THREADS - 1];



//ˮƽ������simd
void* threadFunc(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{
		if (t_id == 0)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//ѭ����������
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//��ȥ
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;

		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(NULL);
}




//ˮƽ������simd
void* threadFunc_horizontal1(void* param)
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		vt = vmovq_n_f32(A[k][k]);

		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				va = vld1q_f32(&(A[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(A[k][j]), va);
			}

			for (; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//ѭ����������
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//��ȥ
			vaik = vmovq_n_f32(A[i][k]);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				vakj = vld1q_f32(&(A[k][j]));
				vaij = vld1q_f32(&(A[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for (; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(NULL);
}

//ˮƽ�黮����simd
void* threadFunc_horizontal2(void* param)
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		vt = vmovq_n_f32(A[k][k]);

		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				va = vld1q_f32(&(A[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(A[k][j]), va);
			}

			for (; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}



		int each = (n - k - 1) / NUM_THREADS;
		int end = 0;
		if (t_id == NUM_THREADS - 1)
		{
			end = n;
		}
		else
		{
			end = k + 1 + each * (t_id + 1);
		}

		//ѭ����������
		for (int i = k + 1 + t_id * each; i < end; i++)
		{
			//��ȥ
			vaik = vmovq_n_f32(A[i][k]);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				vakj = vld1q_f32(&(A[k][j]));
				vaij = vld1q_f32(&(A[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for (; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(NULL);
}




//��ֱ�黮����simd
void* threadFunc_vertical1(void* param)
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		vt = vmovq_n_f32(A[k][k]);

		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				va = vld1q_f32(&(A[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(A[k][j]), va);
			}

			for (; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//ѭ����������
		for (int i = k + 1; i < n; i++)
		{
			//��ȥ
			vaik = vmovq_n_f32(A[i][k]);
			int j;

			int each = (n - k - 1) / NUM_THREADS;
			int end = 0;
			if (t_id == NUM_THREADS - 1)
			{
				end = n;
			}
			else
			{
				end = k + 1 + each * (t_id + 1);
			}


			for (j = k + 1 + t_id * each; j + 4 <= end; j += 4)
			{
				vakj = vld1q_f32(&(A[k][j]));
				vaij = vld1q_f32(&(A[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for (; j < end; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(NULL);
}


//��ֱ���廮����simd
void* threadFunc_vertica2(void* param)
{

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		if (t_id == 0)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//ѭ����������
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1 + t_id; j < n; j += NUM_THREADS)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(NULL);
	return NULL;
}














int main()
{
	for (int i = 0; i < 6; i++)
	{
		cin >> n;
		A_init();
		timespec_get(&start, TIME_UTC);

		//��ʼ���ź���
		sem_init(&sem_leader, 0, 0);

		for (int i = 0; i < NUM_THREADS - 1; ++i)
		{
			sem_init(sem_Divsion, 0, 0);
			sem_init(sem_Elimination, 0, 0);
		}

		//�����߳�
		pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
		threadParam_t* param = new threadParam_t[NUM_THREADS];// ������Ӧ���߳����ݽṹ
		for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		{
			param[t_id].t_id = t_id;
			pthread_create(&handles[t_id], NULL, threadFunc_vertical1, (void*)&param[t_id]);
		}


		for (int t_id = 0; t_id < NUM_THREADS; t_id++)
			pthread_join(handles[t_id], NULL);

		//���������ź���
		sem_destroy(&sem_leader);
		sem_destroy(sem_Divsion);
		sem_destroy(sem_Elimination);

		timespec_get(&end1, TIME_UTC);
		total_duration_sec = end1.tv_sec - start.tv_sec;
		total_duration_nsec = end1.tv_nsec - start.tv_nsec;
		if (total_duration_nsec < 0)
		{
			total_duration_sec--;
			total_duration_nsec += 1000000000L;
		}
		printf("Time of ckYS : %lld.%09lds\n", total_duration_sec, total_duration_nsec);

		deleteA();
	}



}
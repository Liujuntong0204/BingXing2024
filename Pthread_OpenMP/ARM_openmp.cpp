#include <omp.h>
#include <iostream>
#include <sys/time.h>
# include <arm_neon.h> // use Neon
using namespace std;

int n = 1000;
float** A;
const int NUM_THREADS = 4; //工作线程数量
//定义计时变量
struct timespec start;
struct timespec end1;
time_t total_duration_sec;
long total_duration_nsec;


void A_init() {     //未对齐的数组的初始化
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

//串行算法
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


//无SIMD+openmp_static
void f_omp_static()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

//无SIMD+openmp_dynamic
void f_omp_dynamic()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(dynamic, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

//无SIMD+openmp_guided
void f_omp_guided()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(guided, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}



//SIMD+openmp_static
void f_omp_static_neon()
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);

#pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float32x4_t vt = vmovq_n_f32(A[k][k]);
			int j;
			for (j = k + 1; j < n; j++)
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

		//并行部分
#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
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
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}

			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}




//SIMD+openmp_dynamic
void f_omp_dynamic_neon()
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);

#pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float32x4_t vt = vmovq_n_f32(A[k][k]);
			int j;
			for (j = k + 1; j < n; j++)
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

		//并行部分
#pragma omp for schedule(dynamic, 14)
		for (int i = k + 1; i < n; i++)
		{
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
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}

			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}


//SIMD+openmp_guide
void f_omp_guide_neon()
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);

#pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float32x4_t vt = vmovq_n_f32(A[k][k]);
			int j;
			for (j = k + 1; j < n; j++)
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

		//并行部分
#pragma omp for schedule(guided, 1)
		for (int i = k + 1; i < n; i++)
		{
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
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}

			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}


void cal(void(*func)()) {
	//更新数组
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

	timespec_get(&start, TIME_UTC);
	func();
	timespec_get(&end1, TIME_UTC);

}



int main()
{
	cin >> n;
	A = new float* [n];
	for (int i = 0; i < n; i++) {
		A[i] = new float[n];
	}

	// //串行算法
	// cal(f_ordinary);
	// total_duration_sec = end1.tv_sec - start.tv_sec;
	// total_duration_nsec = end1.tv_nsec - start.tv_nsec;
	// if (total_duration_nsec < 0)
	// {
	// 	total_duration_sec--;
	// 	total_duration_nsec += 1000000000L;
	// }
	// printf("Time of serial : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
	// deleteA();

	// //无simd的openmp_static
	// cal(f_omp_static);
	// total_duration_sec = end1.tv_sec - start.tv_sec;
	// total_duration_nsec = end1.tv_nsec - start.tv_nsec;
	// if (total_duration_nsec < 0)
	// {
	// 	total_duration_sec--;
	// 	total_duration_nsec += 1000000000L;
	// }
	// printf("Time of NoSIMD+openmp_static : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
	// deleteA();

	// //无simd的openmp_dynamic
	// cal(f_omp_dynamic);
	// total_duration_sec = end1.tv_sec - start.tv_sec;
	// total_duration_nsec = end1.tv_nsec - start.tv_nsec;
	// if (total_duration_nsec < 0)
	// {
	// 	total_duration_sec--;
	// 	total_duration_nsec += 1000000000L;
	// }
	// printf("Time of NoSIMD+openmp_dynamic : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
	// deleteA();

	// //无SIMD的openmp_guided
	// cal(f_omp_guided);
	// total_duration_sec = end1.tv_sec - start.tv_sec;
	// total_duration_nsec = end1.tv_nsec - start.tv_nsec;
	// if (total_duration_nsec < 0)
	// {
	// 	total_duration_sec--;
	// 	total_duration_nsec += 1000000000L;
	// }
	// printf("Time of NoSIMD+openmp_guided : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
	// deleteA();

	//SIMD+openmp_static
	cal(f_omp_static_neon);
	total_duration_sec = end1.tv_sec - start.tv_sec;
	total_duration_nsec = end1.tv_nsec - start.tv_nsec;
	if (total_duration_nsec < 0)
	{
		total_duration_sec--;
		total_duration_nsec += 1000000000L;
	}
	printf("Time of SIMD+openmp_static : %lld.%09lds\n", total_duration_sec, total_duration_nsec);

	//SIMD+openmp_dynamic
	cal(f_omp_dynamic_neon);
	total_duration_sec = end1.tv_sec - start.tv_sec;
	total_duration_nsec = end1.tv_nsec - start.tv_nsec;
	if (total_duration_nsec < 0)
	{
		total_duration_sec--;
		total_duration_nsec += 1000000000L;
	}
	printf("Time of SIMD+openmp_dynamic : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
	//deleteA();

	//SIMD+openmp_guided
	cal(f_omp_guide_neon);
	total_duration_sec = end1.tv_sec - start.tv_sec;
	total_duration_nsec = end1.tv_nsec - start.tv_nsec;
	if (total_duration_nsec < 0)
	{
		total_duration_sec--;
		total_duration_nsec += 1000000000L;
	}
	printf("Time of SIMD+openmp_guided : %lld.%09lds\n", total_duration_sec, total_duration_nsec);
	//deleteA();
}




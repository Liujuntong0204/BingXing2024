#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <immintrin.h>
#include <time.h>
using namespace std;

const int Num = 1000;
const int pasNum = 10000;
const int lieNum = 30000;

unsigned int Act[lieNum][Num] = { 0 };
unsigned int Pas[lieNum][Num] = { 0 };


void init_A()
{
    unsigned int a;
    ifstream infile("消元子.txt");
    char fin[10000] = { 0 };
    int index;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;
        while (line >> a)
        {
            if (biaoji == 0)
            {
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;
        }
    }
}

void init_P()
{
    unsigned int a;
    ifstream infile("被消元行.txt");
    char fin[10000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;
        while (line >> a)
        {
            if (biaoji == 0)
            {
                Pas[index][Num] = a;
                biaoji = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Pas[index][Num - 1 - j] += temp;
        }
        index++;
    }
}



void f_ordinary()
{
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];
                    Act[index][Num] = 1;
                    break;
                }

            }
        }
    }

    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];
                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}



__m128 va_Pas;
__m128 va_Act;


void f_sse()
{
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {
                    int k;
                    for (k = 0; k + 4 <= Num; k += 4)
                    {
                        va_Pas = _mm_loadu_ps((float*)&(Pas[j][k]));
                        va_Act = _mm_loadu_ps((float*)&(Act[index][k]));

                        va_Pas = _mm_xor_ps(va_Pas, va_Act);
                        _mm_store_ss((float*)&(Pas[j][k]), va_Pas);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {
                    int k;
                    for (k = 0; k + 4 <= Num; k += 4)
                    {
                        va_Pas = _mm_loadu_ps((float*)&(Pas[j][k]));
                        va_Act = _mm_loadu_ps((float*)&(Act[i][k]));
                        va_Pas = _mm_xor_ps(va_Pas, va_Act);
                        _mm_store_ss((float*)&(Pas[j][k]), va_Pas);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }

}


__m256 va_Pas2;
__m256 va_Act2;

void f_avx256()
{
    for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {
                    int k;
                    for (k = 0; k + 8 <= Num; k += 8)
                    {
                        va_Pas2 = _mm256_loadu_ps((float*)&(Pas[j][k]));
                        va_Act2 = _mm256_loadu_ps((float*)&(Act[index][k]));

                        va_Pas2 = _mm256_xor_ps(va_Pas2, va_Act2);
                        _mm256_storeu_ps((float*)&(Pas[j][k]), va_Pas2);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (int i = lieNum % 8 - 1; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {
                    int k;
                    for (k = 0; k + 8 <= Num; k += 8)
                    {
                        va_Pas2 = _mm256_loadu_ps((float*)&(Pas[j][k]));
                        va_Act2 = _mm256_loadu_ps((float*)&(Act[i][k]));
                        va_Pas2 = _mm256_xor_ps(va_Pas2, va_Act2);
                        _mm256_storeu_ps((float*)&(Pas[j][k]), va_Pas2);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}







int main()
{
    struct timespec start1, end1, start2, end2;
    time_t duration_sec1, duration_sec2;
    long duration_nsec1, duration_nsec2;
    time_t total_duration_sec;
    long total_duration_nsec;
    init_A();
    init_P();
    timespec_get(&start1, TIME_UTC);
    //f_ordinary();
    //f_sse();
    f_avx256();
    timespec_get(&end1, TIME_UTC);
    duration_sec1 = end1.tv_sec - start1.tv_sec;
    duration_nsec1 = end1.tv_nsec - start1.tv_nsec;
    if (duration_nsec1 < 0)
    {
        duration_sec1--;
        duration_nsec1 += 1000000000L;
    }
    printf("Time of elimination: %lld.%09lds\n", duration_sec1, duration_nsec1);


}

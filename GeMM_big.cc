/*
GEMM implementations from https://github.com/jaopaulolc/KernelFaRer/tree/main
 * */
#include <cstdlib>
#include <iostream>
//#include <chrono>

#include <random>
#include <unistd.h>
#include <cstring>
#include <functional>


//#define GEM5
#ifdef GEM5
#include <gem5/m5ops.h>
#else
#include <chrono>
#endif

#ifndef CACHE_SIZE_IN_KB
#define CACHE_SIZE_IN_KB (48*1024*1024)
#endif
#ifndef M_DIM
#define M_DIM (2048)
#endif
#ifndef K_DIM
#define K_DIM (2048)
#endif
#ifndef N_DIM
#define N_DIM (2048)
#endif

using namespace std;

void step0_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  // Naive GEMM
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (p = 0; p < k; p++) {
        C[ j * m + i ] += A[ p * m + i ] * B[ j * k + p ];
      }
    }
  }
}

void step1_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  // Naive + Transpose(A) GEMM
  for (i = 0; i < m; i++) {
    const auto A_ptr = &A [ i * k /*+ p*/ ];
    for (j = 0; j < n; j++) {
      const auto C_ptr = &C [ j * m /*+ i*/ ];
      const auto B_ptr = &B [ j * k /*+ p*/];
      for (p = 0; p < k; p++) {
        C_ptr[i] += A_ptr[p] * B_ptr[p];
      }
    }
  }
}

void step2_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  // Naive + Interchange (i,j,p) -> (j,p,i) GEMM
  for (j = 0; j < n; j++) {
    auto B_ptr = &B [ j * k /*+ p*/ ];
    auto C_ptr = &C [ j * m /*+ i*/ ];
    for (p = 0; p < k; p++) {
      auto A_ptr = &A [ p * m /*+ i*/ ];
      for (i = 0; i < m; i++) {
        C_ptr[i] += A_ptr[i] * B_ptr[p];
      }
    }
  }
}

void step3_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  constexpr int i_blk = 256;
  constexpr int j_blk = 256;
  constexpr int p_blk = 256;

  // Naive + Interchange + Blocking GEMM
  for (j = 0; j < n; j+=j_blk) {
    for (p = 0; p < k; p+=p_blk) {
      for (i = 0; i < m; i+=i_blk) {
        const int jj_end = std::min(n, j + j_blk);
        for (long jj = j; jj < jj_end; jj++ ) {
          auto B_ptr = &B [ jj * k /*+ pp*/ ];
          auto C_ptr = &C [ jj * m /*+ ii*/ ];
          const int pp_end = std::min(k, p + p_blk);
          for (long pp = p; pp < pp_end; pp++) {
            auto A_ptr = &A [ pp * m /*+ ii*/ ];
            const int ii_end = std::min(m, i + i_blk);
            for (long ii = i; ii < ii_end; ii++) {
              C_ptr[ii] += A_ptr[ii] * B_ptr[pp];
            }
          }
        }
      }
    }
  }
}

void step4_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  constexpr int i_blk = 256;
  constexpr int j_blk = 256;
  constexpr int p_blk = 256;

#define pack_matrix(Mat_pack, Mat, t, t_blk, s_blk, S) \
  for (auto _s = 0,pack_it=0; _s < S; _s+=s_blk) \
    for (auto _t = t; _t < t+t_blk; _t++) { \
      const double *Mat_ptr = &Mat[_t*S]; \
      for (auto _ss = _s; _ss < _s+s_blk; _ss+=8) { \
        Mat_pack[pack_it + 0] = Mat_ptr[_ss + 0]; \
        Mat_pack[pack_it + 1] = Mat_ptr[_ss + 1]; \
        Mat_pack[pack_it + 2] = Mat_ptr[_ss + 2]; \
        Mat_pack[pack_it + 3] = Mat_ptr[_ss + 3]; \
        Mat_pack[pack_it + 4] = Mat_ptr[_ss + 4]; \
        Mat_pack[pack_it + 5] = Mat_ptr[_ss + 5]; \
        Mat_pack[pack_it + 6] = Mat_ptr[_ss + 6]; \
        Mat_pack[pack_it + 7] = Mat_ptr[_ss + 7]; \
        pack_it += 8; \
      } \
    }

  double *A_pack = (double*)malloc(sizeof(double)*p_blk*m);
  double *B_pack = (double*)malloc(sizeof(double)*j_blk*k);

  // Naive + Interchange + Blocking + Packing GEMM
  for (j = 0; j < n; j+=j_blk) {
    int blockB;
    pack_matrix(B_pack, B, j, j_blk, p_blk, k);
    for (p = 0, blockB = 0; p < k; p+=p_blk, blockB++) {
      int blockA;
      pack_matrix(A_pack, A, p, p_blk, i_blk, m);
      for (i = 0, blockA = 0; i < m; i+=i_blk, blockA++) {
        const int jj_end = std::min(n, j + j_blk);
        for (long jj = j, pack_j=0; jj < jj_end; jj++, pack_j++) {
          auto B_ptr = &B_pack [ blockB * (j_blk * p_blk) + pack_j * p_blk];
          auto C_ptr = &C [ jj * m /*+ ii*/ ];
          const int pp_end = std::min(k, p + p_blk);
          for (long pp = p, pack_p=0; pp < pp_end; pp++, pack_p++) {
            auto A_ptr = &A_pack [ blockA * (p_blk * i_blk) + pack_p * i_blk ];
            const int ii_end = std::min(m, i + i_blk);
            for (long ii = i, pack_i=0; ii < ii_end; ii++, pack_i++) {
              C_ptr[ii] += A_ptr[pack_i] * B_ptr[pack_p];
            }
          }
        }
      }
    }
  }
  free(A_pack);
  free(B_pack);
}

void randomizeMatrix(double *mat, int d1, int d2) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<uint64_t> dist(0, 2);

  for (int i = 0; i < d1 * d2; ++i) {
    mat[i] = static_cast<double>(dist(mt));
  }
}

typedef std::function<void(double*,double*,double*,int,int,int)> GEMM_FUNC;

void runGeMMTest(const char* name, GEMM_FUNC f, double *A, double *B, double *C, double *CExp, int m, int k,
    int n) {
  randomizeMatrix(A, m, k);
  randomizeMatrix(B, k, n);
  memset(C, 0, sizeof(double)*m*n);
  memset(CExp, 0, sizeof(double)*m*n);

#ifdef GEM5
  m5_reset_stats(0,0);
#else 
  auto t0 = chrono::high_resolution_clock::now();
#endif
  f(A, B, C, m, k, n);
#ifdef GEM5
  m5_dump_stats(0,0);
  cerr << name;
#else
  auto t1 = chrono::high_resolution_clock::now();
  cerr << name << " "
       << chrono::duration_cast<chrono::milliseconds>(t1-t0).count()
       << " ms";
#endif


  cerr << '\n';
}

int main(int argc, char **argv) {

  double *A[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  double *B[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  double *C[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  double *CExp = nullptr;
#define check_alloc(a) \
  if (a != 0) { \
    perror("posix_memalign"); \
    exit(-1); \
  }
  const long PAGE_SIZE = sysconf(_SC_PAGESIZE);
  for(int i = 0; i < 5; ++i) {
    check_alloc(posix_memalign((void**)&(A[i]),    PAGE_SIZE, sizeof(double) * M_DIM * K_DIM));
    check_alloc(posix_memalign((void**)&(B[i]),    PAGE_SIZE, sizeof(double) * K_DIM * N_DIM));
    check_alloc(posix_memalign((void**)&(C[i]),    PAGE_SIZE, sizeof(double) * M_DIM * N_DIM));
  }
  check_alloc(posix_memalign((void**)&CExp, PAGE_SIZE, sizeof(double) * M_DIM * N_DIM));

#if defined(WARMUP)
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      M_DIM, N_DIM, K_DIM, 1.0, A, M_DIM, B, K_DIM, 1.0, C, M_DIM);
#endif
  cerr << "Starting benchmarks\n";
  runGeMMTest("Step0", step0_dgemm, A[0], B[0], C[0], CExp, M_DIM, K_DIM, N_DIM);
  runGeMMTest("Step1", step1_dgemm, A[1], B[1], C[1], CExp, M_DIM, K_DIM, N_DIM);
  runGeMMTest("Step2", step2_dgemm, A[2], B[2], C[2], CExp, M_DIM, K_DIM, N_DIM);
  runGeMMTest("Step3", step3_dgemm, A[3], B[3], C[3], CExp, M_DIM, K_DIM, N_DIM);
  runGeMMTest("Step4", step4_dgemm, A[4], B[4], C[4], CExp, M_DIM, K_DIM, N_DIM);

  for(int i = 0; i < 5; ++i) {
    free(A[i]);
    free(B[i]);
    free(C[i]);
  }
  free(CExp);

  return 0;
}

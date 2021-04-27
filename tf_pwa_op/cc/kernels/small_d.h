// kernel_example.h
#ifndef KERNEL_TIME_TWO_H_
#define KERNEL_TIME_TWO_H_
#include<complex>
using namespace std;
namespace tensorflow {

namespace functor {

template <typename Device, typename T> struct SmallDFunctor {
  void operator()(const Device &d, int size, int j, const T *in, const T *w,
                  T *sincos, T *out);
};

template <typename Device, typename T> struct DeltaDFunctor {
  void operator()(const Device &d, int size, int j, const T* small_d, 
        const T* alpha, 
        const T* gamma,
        const int* la,
        const int* lb,
        const int* lc,
        T* out_r,
        T* out_i,
        int na, int nb, int nc
        );
};

} // namespace functor

} // namespace tensorflow

#endif // KERNEL_TIME_TWO_H_

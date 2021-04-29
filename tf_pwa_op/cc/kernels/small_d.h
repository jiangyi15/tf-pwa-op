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

template <typename Device, typename T> struct MonmentLambdaFunctor {
  void operator()(const Device &d, int size, const T* m0, 
        const T* m1,
        const T* m2, T* out);
};

template <typename Device, typename T> struct  MonmentLambdaGradFunctor {
  void operator()(const Device &d, int size, const T* m0, 
        const T* m1,
        const T* m2, T*out);
};


template <typename Device, typename T> struct BlattWeisskopfFunctor {
  void operator()(const Device &device,const int size,const int nl,const float d, const int* l, 
        const T* q,
        const T* q0, T* out);
};


} // namespace functor

} // namespace tensorflow

#endif // KERNEL_TIME_TWO_H_

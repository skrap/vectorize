#pragma once

#include <vectorize/arch.h>
#include <cmath>
#include <algorithm>

namespace vectorize {

  template <unsigned N>
  struct Placeholder {
    template <class T, unsigned M>
    T apply(T (&x)[M]) const {
      static_assert(N < M,
                    "Too few arguments to placeholder expression,"
                    "this is probably due to passing an expression"
                    "with more placeholders than actual inputs.");
      return x[N];
    }
  };

  class Constant {
  public:
    Constant(float value) : _value(value) {}
    template <unsigned N>
      arch::float32x4_t apply(arch::float32x4_t(&)[N]) const { return arch::vec_set(_value); }
    template <unsigned N>
    float apply(float(&)[N]) const { return _value; }
  private:
    float _value;
  };

  template <class L, class R, class Op>
  class BinOp {
  public:
    BinOp(const L& l, const R& r) : _l(l), _r(r) {}

    template <class T, unsigned N>
    T apply(T (&x)[N]) const {
      return Op::eval(_l.apply(x), _r.apply(x));
    }
  private:
    const L _l;
    const R _r;
  };

  struct AddOp {
    static arch::float32x4_t eval(arch::float32x4_t x, arch::float32x4_t y) {
        return arch::vec_add(x, y);
    }
    static float eval(float x, float y) {
      return x + y;
    }
  };

  struct SubOp {
    static arch::float32x4_t eval(arch::float32x4_t x, arch::float32x4_t y) {
        return arch::vec_sub(x, y);
    }
    static float eval(float x, float y) {
      return x - y;
    }
  };

  struct MulOp {
    static arch::float32x4_t eval(arch::float32x4_t x, arch::float32x4_t y) {
        return arch::vec_mul(x, y);
    }
    static float eval(float x, float y) {
      return x * y;
    }
  };

  struct MaxOp {
    static arch::float32x4_t eval(arch::float32x4_t x, arch::float32x4_t y) {
        return arch::vec_max(x, y);
    }
    static float eval(float x, float y) {
      return std::max(x, y);
    }
  };

  struct MinOp {
    static arch::float32x4_t eval(arch::float32x4_t x, arch::float32x4_t y) {
        return arch::vec_min(x, y);
    }
    static float eval(float x, float y) {
      return std::min(x, y);
    }
  };

  template <class T>
  class Expr {
  public:
    Expr(const T& t) : _t(t) {}
    Expr() : _t() {}

    template <unsigned N>
    arch::float32x4_t apply(arch::float32x4_t (&x)[N]) const {
      return _t.apply(x);
    }
    template <unsigned N>
    float apply(float (&x)[N]) const {
      return _t.apply(x);
    }
  private:
    const T _t;
  };

  const Expr<Placeholder<0>> _x;
  const Expr<Placeholder<1>> _y;

  // Macro for generating binary operators. OpFunction is the C++ function
  // the user calls, i.e. operator+ or max. BinOpClass is the expression
  // template class that implements the operator.
#define GENERATE_BINARY_OPERATOR(OpFunction, BinOpClass)              \
  template <class L, class R>                                         \
  Expr<BinOp<Expr<L>, Expr<R>, BinOpClass>>                           \
  OpFunction(const Expr<L>& l, const Expr<R>& r) {                    \
    typedef BinOp<Expr<L>, Expr<R>, BinOpClass> binop_type;           \
    return Expr<binop_type>(binop_type(l, r));                        \
  }                                                                   \
  template <class L>                                                  \
  Expr<BinOp<Expr<L>, Expr<Constant>, BinOpClass>>                    \
  OpFunction(const Expr<L>& l, float r) {                             \
    typedef BinOp<Expr<L>, Expr<Constant>, BinOpClass> binop_type;    \
    Expr<Constant> c= Constant(r);                                    \
    return Expr<binop_type>(binop_type(l, c));                        \
  }                                                                   \
  template <class R>                                                  \
  Expr<BinOp<Expr<Constant>, Expr<R>, BinOpClass>>                    \
  OpFunction(float l, const Expr<R>& r) {                             \
    typedef BinOp<Expr<Constant>, Expr<R>, BinOpClass> binop_type;    \
    Expr<Constant> c= Constant(l);                                    \
    return Expr<binop_type>(binop_type(c, r));                        \
  }                                                                   \

  GENERATE_BINARY_OPERATOR(operator+, AddOp)
  GENERATE_BINARY_OPERATOR(operator-, SubOp)
  GENERATE_BINARY_OPERATOR(operator*, MulOp)
  GENERATE_BINARY_OPERATOR(max, MaxOp)
  GENERATE_BINARY_OPERATOR(min, MinOp)

#undef GENERATE_BINARY_OPERATOR

  template <class T, class Op>
  class UnaryOp {
  public:
    UnaryOp(const T& t) : _t(t) {}

    template <unsigned N>
    arch::float32x4_t apply(arch::float32x4_t (&x)[N]) const {
      return Op::eval(_t.apply(x));
    }
    template <unsigned N>
    float apply(float (&x)[N]) const {
      return Op::eval(_t.apply(x));
    }
  private:
    const T _t;
  };

  struct AbsOp {
    static arch::float32x4_t eval(arch::float32x4_t x) {
        return arch::vec_abs(x);
    }
    static float eval(float x) {
      return std::abs(x);
    }
  };

  struct SqrtOp {
    static arch::float32x4_t eval(arch::float32x4_t x) {
        return arch::vec_sqrt(x);
    }
    static float eval(float x) {
      return std::sqrt(x);
    }
  };

  // Macro for generating unary operators. OpFunction is the C++ function
  // the user calls, i.e. abs or sqrt. UnOpClass is the expression
  // template class that implements the operator.
#define GENERATE_UNARY_OPERATOR(OpFunction, UnOpClass)              \
  template <class T>                                                \
  Expr<UnaryOp<Expr<T>, UnOpClass>> OpFunction(const Expr<T>& t) {  \
    typedef UnaryOp<Expr<T>, UnOpClass> unop_type;                  \
    return Expr<unop_type>(unop_type(t));                           \
  }

  GENERATE_UNARY_OPERATOR(abs, AbsOp)
  GENERATE_UNARY_OPERATOR(sqrt, SqrtOp)

#undef GENERATE_UNARY_OPERATOR

  template <class F>
  void apply(unsigned n, const float* src, float* target, F f) {
    for (; n>=4; n-=4, src+=4, target+=4) {
        arch::float32x4_t x[]= { arch::vec_load(src) };
        arch::vec_store(target, f.apply(x));
    }

    for (; n>0; --n, ++src, ++target) {
      float x[]= { *src };
      *target= f.apply(x);
    }
  }

  template <class F>
  void apply2(unsigned n, const float* src1, const float* src2,
              float* target, F f) {
    for (; n>=4; n-=4, src1+=4, src2+=4, target+=4) {
        arch::float32x4_t x[]= { arch::vec_load(src1), arch::vec_load(src2) };
        arch::vec_store(target, f.apply(x));
    }

    for (; n>0; --n, ++src1, ++src2, ++target) {
      float x[]= { *src1, *src2 };
      *target= f.apply(x);
    }
  }

}


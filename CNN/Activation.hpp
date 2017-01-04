#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cfloat>

#ifndef __NN_ACTIVATION__
#define __NN_ACTIVATION__

typedef void (*activation)(double*, int);

enum ACTIVATION_TYPE {
  SIGMOID = 0,
  TANH = 1,
  RECTIFIER = 2,
  ORIGINAL = 3
};

class Activation {

public:
  static activation getActivation(int type) {

    static activation f[] = {Activation::sigmoid, Activation::tanh, Activation::rectifier, Activation::original};
    return f[type];
  }
  static activation getDActivation(int type) {

    static activation df[] = {Activation::dsigmoid, Activation::dtanh, Activation::drectifier, Activation::doriginal};
    return df[type];
  }


  static void sigmoid(double *a, int n) {

    for(int i=0;i<n;i++) {
      a[i] = 1.0 / ( exp(-a[i]) + 1.0 );
    }
  }
  static void dsigmoid(double *a, int n) {

    for(int i=0;i<n;i++) {
      a[i] = a[i] * ( 1.0 - a[i] );
    }
  }

  static void tanh(double *a, int n) {

    for(int i=0;i<n;i++) {
      double x1 = exp(a[i]);
      double x2 = exp(-a[i]);
       a[i] = (x1 - x2) / (x1 + x2);
    }
  }

  static void dtanh(double *a, int n) {

    for(int i=0;i<n;i++) {
      a[i] = 1 - a[i]*a[i];
    }
  }

  static void rectifier(double *a, int n) {

    for(int i=0;i<n;i++) {
      if(a[i] < 0)
          a[i] = 0;
    }
  }

  static void drectifier(double *a, int n) {
    for(int i=0;i<n;i++) {
      a[i] = (a[i] > 0) ? 1 : 0;
    }
  }

  static void original(double *a, int n) {
  }

  static void doriginal(double *a, int n) {
    for(int i=0;i<n;i++) {
      a[i] = 1;
    }
  }

};

#endif

#include "FLayer.hpp"

#ifndef __NN_SM_LAYER__
#define __NN_SM_LAYER__

class SoftmaxLayer : public FLayer {


public:
	SoftmaxLayer(Layer *prev=NULL) : FLayer(prev) {

		this->_layer_type = SOFTMAX_LAYER;
	}
	SoftmaxLayer(int n, Layer *prev, Layer *next = NULL) :FLayer(n, SIGMOID, prev, next) {

		this->_layer_type = SOFTMAX_LAYER;

	}
	void forward() {
 		// [n, np]*[np, 1] + [n, 1]
		//_u_a = _u_W * _prev->getActivation() + _u_b;
		int n = _unit_count, np = _prev_unit_count;
		memcpy(_u_a, _u_b, n*sizeof(double));

		double *pa = _prev->getActivation();
		for(int i=0;i<n;i++) {
			double d = 0;
			for(int j=0;j<np;j++) {
				d += _u_W[i*np+j] * pa[j];
			}
			_u_a[i] += d;
		}
    double sum = 0;
    double maxv = _u_a[0];
    for(int i=1;i<n;i++) {
      if(_u_a[i] > maxv)
        maxv = _u_a[i];
    }

	  for(int i=0;i<n;i++) {
      sum += exp(_u_a[i]-maxv);
    }
    for(int i=0;i<n;i++) {
      _u_a[i] = exp(_u_a[i]-maxv) / sum;
    }
	}

};


#endif

#include <cmath>
#include <fstream>
#include <memory.h>
#include <cstdlib>
#include <cstdio>
#include <cfloat>

#include "Activation.hpp"

#ifndef __NN_LAYER__
#define __NN_LAYER__
#define MIN(a,b) ((a)<(b)?(a):(b))

class Layer {

protected:
	int _layer_id;

	Layer* _prev;
	Layer* _next;

	double* _u_a;
	double* _u_delta;
	double* _u_W;
	double* _u_b;


	int _unit_count;
	int _prev_unit_count;
	int _width;
	int _height;
	int _map_num;
	int _layer_type;

	activation _act_f;
	activation _d_act_f;
public:
	enum LAYER_TYPE {
		DEFAULT_LAYER = 1,
		INPUT_LAYER ,
		MAX_POOLING_LAYER,
		AVG_POOLING_LAYER,
		FWS_CONV_LAYER,
		PWS_CONV_LAYER,
		FULL_LAYER,
		SOFTMAX_LAYER,
		RANGE_LAYER
	};
	Layer(Layer *prev, Layer *next) {

		this->_prev = prev;
		this->_next = next;

		_width = 0;
		_height = 0;
		_unit_count = 0;
		_prev_unit_count = 0;
		_map_num = 0;
		_layer_type = DEFAULT_LAYER;

		_u_a = NULL;
		_u_delta = NULL;
		_u_W = NULL;
		_u_b = NULL;
	}
	virtual ~Layer() {
		clear();
	}

	void clear() {
		if(_u_a)
			delete [] _u_a;
		if(_u_delta)
			delete [] _u_delta;
		if(_u_W)
			delete [] _u_W;
		if(_u_b)
			delete [] _u_b;
	}
	virtual void write(std::ofstream &fout) = 0;
	virtual void read(std::ifstream &fin) = 0;

	Layer *getPrevLayer() {
		return _prev;
	}
	Layer *getNextLayer() {
		return _next;
	}
	void setNextLayer(Layer *l) {
		_next = l;
	}
	void setPrevLayer(Layer *l) {
		_prev = l;
	}

	int getWidth() {
		return _width;
	}
	int getHeight() {
		return _height;
	}
	int getMapNum() {
		return _map_num;
	}
	int getUnitCount() {
		return _unit_count;
	}
	double* getActivation() {
		return _u_a;
	}
	double* getDelta() {
		return _u_delta;
	}



	virtual void init() = 0;
	virtual void updateDelta() = 0;
	virtual void backpropagation() = 0;
	virtual void forward() = 0;
	virtual void updateParameters(int,double,double,double) = 0;
	virtual int getTotalUnitCount() = 0;

	void setDelta(double *a, int n) {

		memcpy(_u_delta, a, sizeof(double)*n);
	}
	double* getWeights() {
		return _u_W;
	}
};


#endif

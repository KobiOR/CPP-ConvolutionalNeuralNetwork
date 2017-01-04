

#include "Layer.hpp"
#include "InputLayer.hpp"
#include "FLayer.hpp"
#include "JointLayer.hpp"
#include "FWSConvLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "AvgPoolingLayer.hpp"
#include "PWSConvLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "RangeLayer.hpp"
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cfloat>
#include <ctime>
#include <cassert>

#ifndef __NN_SPARROW__
#define __NN_SPARROW__


class Network {


protected:

	void (*_call_back)(void*);
	std::vector<Layer*> _layers;
	std::vector<InputLayer*> _inputlayers;
	double _learning_decay_rate;
	double _learning_rate;
	double _weight_decay_parameter;
	double _error_bound;
	double _avg_error;
	int _epoch_count;
	double _momentum;
	int _train_batch_count;
	clock_t _run_time;
	bool _ready;

public:
	Network() {
		_momentum = 0.9;
		_learning_rate = 0.01;
		_learning_decay_rate = 0.9;
		_weight_decay_parameter = 0.0001;
		_error_bound = 0.00001;
		_epoch_count = 20;
		_train_batch_count = 10;
		_avg_error = 0;
		_ready = false;
		_call_back = NULL;
		_run_time = clock();

	}
	~Network() {
		reset();
	}
	clock_t getRunTime() {
		return _run_time;
	}
	void setCallbackFunction(void (*f)(void*)) {
		this->_call_back = f;
	}
	void setErrorBound(double err) {
		_error_bound = err;
	}
	void setEpochCount(int n) {
		_epoch_count = n;
	}
	void setLearningRate(double a) {
		this->_learning_rate = a;
	}
	void setWeightDecay(double d) {
		this->_weight_decay_parameter = d;
	}
	void setTrainBatchCount(int n) {
		this->_train_batch_count = n;
	}
	void setMomentum(double a) {
		this->_momentum = a;
	}
	void setLearningDecayRate(double a) {
		this->_learning_decay_rate = a;
	}
	double getAvgError() {
		return this->_avg_error;
	}
	void reset() {
		while(!_layers.empty()) {
			delete _layers.back();
			_layers.pop_back();
		}
		while(!_inputlayers.empty()) {
			delete _inputlayers.back();
			_inputlayers.pop_back();
		}
		_ready = false;
	}
	int getLayerCount() {
		return _layers.size();
	}

	void addLayer(Layer *l) {

		_layers.push_back(l);
	}
	void save(const char *path) {

		std::ofstream fout(path);
		fout << _momentum << " " << _learning_rate << " " << _learning_decay_rate << " " << _weight_decay_parameter << std::endl;\
		fout << _inputlayers.size() << std::endl;
		for(int i=0;i<_inputlayers.size();i++)
			_inputlayers[i]->write(fout);
		fout << _layers.size() << std::endl;
		for(int i=0;i<_layers.size();i++) {
			_layers[i]->write(fout);
		}
	}

	void load(const char *path) {

		std::ifstream fin(path);
		fin >> _momentum >> _learning_rate >> _learning_decay_rate >> _weight_decay_parameter;
		int n = 0;
		fin >> n;
		for(int i=0;i<n;i++) {
			int type;
			fin >> type;
			InputLayer *l = new InputLayer();
			_inputlayers.push_back(l);
			l->read(fin);
		}

		fin >> n;
		for(int i=0;i<n;i++) {
			int type;
			fin >> type;
			Layer *l;
			switch(type) {
				case Layer::FULL_LAYER:
					l = new FLayer();
					break;
				case Layer::FWS_CONV_LAYER:
					l = new FWSConvLayer();
					break;
				case Layer::PWS_CONV_LAYER:
					l = new PWSConvLayer();
					break;
				case Layer::AVG_POOLING_LAYER:
					l = new AvgPoolingLayer();
					break;
				case Layer::MAX_POOLING_LAYER:
					l = new MaxPoolingLayer();
					break;
				case Layer::SOFTMAX_LAYER:
					l = new SoftmaxLayer();
					break;
				default:
					break;
			}
			if(i > 0) {
				l->setPrevLayer(_layers.back());
			}
			_layers.push_back(l);
			l->read(fin);
		}

		// for(int i=1;i<_layers.size();i++) {
		// 	_layers[i]->setPrevLayer(_layers[i-1]);
		// }
		_layers[0]->setPrevLayer(_inputlayers[0]);
		_ready = true;
	}
	Layer* addInputLayer(int w, int h, int ch) {
		//reset();
		InputLayer *l = new InputLayer(w, h, ch);
		//_layers.push_back(l);
		_inputlayers.push_back(l);
		return l;
	}
	Layer* addFWSConvLayer(Layer* pl, int w, int h, int nm, int at = SIGMOID) {
		// if(_layers.empty())
		// 	return NULL;
		//Layer *pl = _layers.back();
		FWSConvLayer *l = new FWSConvLayer(w, h, nm, at, pl, NULL);
		_layers.push_back((Layer*)l);
		if(pl) {
			pl->setNextLayer((Layer*)l);
		}
		return l;
	}
	Layer* addPWSConvLayer(Layer* pl, int w, int h, int sw, int sh, int nm, int stpx = 1, int stpy = 1, int at = SIGMOID) {
		// if(_layers.empty())
		// 	return NULL;
		//Layer *pl = _layers.back();
		PWSConvLayer *l = new PWSConvLayer(w, h, sw, sh, nm, stpx, stpy, at, pl);
		_layers.push_back((Layer*)l);
		if(pl) {
			pl->setNextLayer((Layer*)l);
		}
		return l;
	}
	Layer* addRangeLayer(Layer* pl, int st, int at = SIGMOID) {

		RangeLayer *l = new RangeLayer(st, at, pl);
		_layers.push_back((Layer*)l);
		if(pl) {
			pl->setNextLayer((Layer*)l);
		}
		return l;
	}
	Layer* addJointLayer(std::vector<Layer*> &ch) {
		JointLayer *l = new JointLayer(ch);
		_layers.push_back((Layer*)l);
		for(int i=0;i<ch.size();i++) {
			ch[i]->setNextLayer((Layer*)l);
		}
		return l;
	}
	Layer* addMaxPoolingLayer(Layer* pl, int w, int h) {

		// if(_layers.empty())
		// 	return NULL;
		//Layer *pl = _layers.back();
		MaxPoolingLayer *l = new MaxPoolingLayer(w, h, pl, NULL);
		_layers.push_back((Layer*)l);
		if(pl) {
			pl->setNextLayer((Layer*)l);
		}
		return l;

	}
	Layer* addAvgPoolingLayer(Layer* pl, int w, int h) {

		// if(_layers.empty())
		// 	return NULL;
		//Layer *pl = _layers.back();
		AvgPoolingLayer *l = new AvgPoolingLayer(w, h, pl, NULL);
		_layers.push_back((Layer*)l);
		if(pl) {
			pl->setNextLayer((Layer*)l);
		}
		return l;

	}
	Layer* addFullLayer(Layer* pl, int n, int at = SIGMOID) {

		// if(_layers.empty())
		// 	return NULL;
		//Layer *pl = _layers.back();

		FLayer *l = new FLayer(n, at, pl, NULL);
		_layers.push_back((Layer*)l);
		if(pl) {
			pl->setNextLayer((Layer*)l);
		}
		return l;
	}
	Layer *addSoftmaxLayer(Layer* pl, int n) {

			// if(_layers.empty())
			// 	return NULL;
			//Layer *pl = _layers.back();

			FLayer *l = new SoftmaxLayer(n, pl, NULL);
			_layers.push_back((Layer *&&) (Layer*)l);
			if(pl) {
				pl->setNextLayer((Layer*)l);
			}
			return l;
	}

	void prepare() {

		for(int i=0;i<(int)_layers.size();i++) {
			_layers[i]->init();
		}
		for(int i=0;i<_inputlayers.size();i++) {
			_inputlayers[i]->init();
		}

	}



	bool train(std::vector<std::vector<double> > &input, std::vector<int> &output){
		if(_inputlayers.size() < 1 || _layers.size() < 1)
			return false;
		if(input.size() <= 0 || input.size() != output.size())
			return false;
		if(_inputlayers.front()->getTotalUnitCount() != input[0].size())
			return false;
		if(!_ready) {
			prepare();
			_ready = true;
		}

		int sz = _layers.size();

		int len = input.size();
		int dim = input[0].size();
		//int odim1 = output[0].size();
		int odim = 0;
		for(int i=0;i<output.size();i++) {
			if(output[i]+1 > odim)
				odim = output[i]+1;
		}
		_avg_error = 100;

		double *ovec = new double[odim];

		//InputLayer *input_layer = (InputLayer*)_layers.front();
		FLayer *output_layer = (FLayer*)_layers.back();

		int *rank = new int[len];
		for(int i=0;i<len;i++)
			rank[i] = i;

		double E = 0;
		unsigned long long itr;
		unsigned long long tot = this->_epoch_count * len;
		for(itr = 0; itr < tot; itr++) {

			int idx = itr % len;
			if(idx == 0) {

				//shuffle is important!!!
				for(int i=0;i<len;i++) {
					int j = i + rand() % (len - i);
					std::swap(rank[i], rank[j]);
				}

				if(itr > 0) {
					E /= len;
					if(itr > 0 && fabs(E-_avg_error) < _error_bound) {
							printf("%lf %lf\n", E, _avg_error );
							break;
					}
					_avg_error = E;
					E = 0;
					_learning_rate *= _learning_decay_rate;
					if(this->_call_back)
						this->_call_back(this);
				}
				_run_time = clock();
			}

			printf("%d\n", idx);
			idx = rank[idx];

			for(int i=0;i<_inputlayers.size();i++)
				_inputlayers[i]->inputSample(&input[idx][0], input[idx].size());

			for(int j=0;j<sz;j++) {
				_layers[j]->forward();
				printf(".");
			}


			memset(ovec, 0, sizeof(double)*odim);
			ovec[output[idx]] = 1;

			output_layer->calculateDelta(ovec, odim);
			double *a = output_layer->getActivation();
			for(int i=0;i<odim;i++) {
				double t = a[i] - ovec[i];
				E += fabs(t);
			}

			for(int j=sz-1;j>=0;j--) {
				_layers[j]->backpropagation();
			}

			if(itr % _train_batch_count == 0) {
				for(int j=sz-1;j>=0;j--) {
					_layers[j]->updateParameters(_train_batch_count, _learning_rate, _weight_decay_parameter, _momentum);
				}
			}
		}

		delete [] rank;
		delete [] ovec;

		return true;
	}

	bool predict(std::vector<double> &input, int &output, double *ovec=NULL) {

		assert(_inputlayers.size() >= 1 && _layers.size() >= 1);
		if(_inputlayers.size() < 1 || _layers.size() < 1)
			return false;
		int dim = input.size();
		assert(_inputlayers.front()->getTotalUnitCount() == dim);
		if(_inputlayers.front()->getTotalUnitCount() != dim)
			return false;

		for(int i=0;i<_inputlayers.size();i++)
			_inputlayers[i]->inputSample(&input[0], dim);


		int sz = _layers.size();
		for(int i=0;i<sz;i++) {
			_layers[i]->forward();
		}


		output = 0;

		double *af = ((FLayer*)_layers.back())->getActivation();
		for(int i=1;i<_layers.back()->getUnitCount();i++) {
			if(af[i] > af[output])
				output = i;
		}

		if(ovec) {
			for(int i=0;i<_layers.back()->getUnitCount();i++)
				ovec[i] = af[i];
		}


		return true;
	}
	void backprop_once(double *ovec, int odim, double conf) {

		SoftmaxLayer *output_layer = (SoftmaxLayer*)_layers.back();
		output_layer->calculateDelta(ovec, odim);
		double *dt = output_layer->getDelta();
		for(int i=0;i<odim;i++) {
			dt[i] *= conf;
		}

		int sz = _layers.size();
		for(int j=sz-1;j>=0;j--) {
			_layers[j]->backpropagation();
		}

		for(int j=sz-1;j>=0;j--) {
			_layers[j]->updateParameters(1, 0.1, 0, 0);
		}
	}
};

#endif

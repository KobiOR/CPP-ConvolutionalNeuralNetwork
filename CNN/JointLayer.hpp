
#include "Layer.hpp"
#include <vector>

#ifndef __NN_JOINT_LAYER__

#define __NN_JOINT_LAYER__

class JointLayer : public Layer {

protected:

	std::vector<Layer*> _children;

public:
	JointLayer(std::vector<Layer*> &ch) :	Layer(NULL, NULL) {

		int tot = 0;

		int sz = ch.size();
		for(int i=0;i<sz;i++) {
			_children.push_back(ch[i]);
			tot += ch[i]->getTotalUnitCount();
		}

		this->_unit_count = tot;
		this->_prev_unit_count = tot;
		this->_width = tot;
		this->_height = 1;
		this->_map_num = 1;

		_u_a = NULL;
		_u_delta = NULL;
	}
	JointLayer() : Layer(NULL, NULL) {

		this->_unit_count = 0;
		this->_prev_unit_count = 0;
		this->_width = 0;
		this->_height = 1;
		this->_map_num = 1;

		_u_a = NULL;
		_u_delta = NULL;

	}

	~JointLayer() {

	}


	void init() {

		//clear();
		int n = this->_unit_count;
		if(n > 0) {
			_u_a = new double[n]; //Eigen::MatrixXd::Zero(n,1);
			memset(_u_a, 0, n*sizeof(double));

			_u_delta = new double[n]; //Eigen::MatrixXd::Zero(n,1);
			memset(_u_delta, 0, n*sizeof(double));
		}

	}

	void addLayer(Layer *l) {

		_children.push_back(l);
	}
	void join() {
		int tot = 0;
		int sz = _children.size();
		for(int i=0;i<sz;i++) {
			tot += _children[i]->getTotalUnitCount();
		}
		this->_unit_count = this->_prev_unit_count = tot;
	}

	void forward() {

		int sh = 0;

		for(int i=0;i<_children.size();i++) {
			double *pa = _children[i]->getActivation();
			int n = _children[i]->getTotalUnitCount();
			memcpy(_u_a+sh, pa, sizeof(double)*n);
			sh += n;
		}

	}
	void backpropagation() {

		int sh = 0;
		for(int i=0;i<_children.size();i++) {
			double *pdt = _children[i]->getDelta();
			int n = _children[i]->getTotalUnitCount();
			memcpy(pdt, _u_delta+sh, sizeof(double)*n);
			sh += n;
		}

	}
	void updateParameters(int m, double alpha, double lambda, double mu) {



	}
	void updateDelta() {


	}

	int getTotalUnitCount() {
		return _unit_count;
	}

	void clear() {
		Layer::clear();
		_children.clear();
	}

	void write(std::ofstream &fout) {


	}
	void read(std::ifstream &fin) {


	}

};


#endif

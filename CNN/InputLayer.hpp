#include "Layer.hpp"


#ifndef __NN_INPUT_LAYER__
#define __NN_INPUT_LAYER__

class InputLayer : public Layer {


public:
  InputLayer() : Layer(NULL, NULL) {

    this->_layer_type = INPUT_LAYER;
  }

  InputLayer(int w, int h, int ch) : Layer(NULL, NULL) {

    this->_unit_count = w*h;
    this->_prev_unit_count = 0;
    this->_map_num = ch;
    this->_width = w;
    this->_height = h;
    this->_layer_type = INPUT_LAYER;
    _u_a = NULL;
  }
  ~InputLayer() {
  }


  void init() {

    Layer::clear();
    _u_a = new double[_unit_count];
    //_u_delta = new double[_unit_count];
  }
  bool inputSample(double *a, int n) {
		if(n != _unit_count)
			return false;
    memcpy(_u_a, a, sizeof(double)*n);
		// for(int i=0;i<n;i++) {
		// 	_u_a[i] = a[i];
		// }
		return true;
	}

  void updateDelta() {

  }
	void backpropagation() {

  }
	void forward() {

  }
	void updateParameters(int,double,double,double) {

  }
  int getTotalUnitCount() {
    return _unit_count;
  }

  void write(std::ofstream &fout) {

    fout << _layer_type << std::endl;
    fout << _unit_count << " " << _prev_unit_count << " ";
    fout << _width << " " << _height << " " << _map_num << std::endl;
  }
  void read(std::ifstream &fin) {

    fin >> _unit_count >> _prev_unit_count;
    fin >> _width >> _height >> _map_num;
    init();
  }

};

#endif

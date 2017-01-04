
#include "Layer.hpp"


#ifndef __NN_FWS_CONV_LAYER__

#define __NN_FWS_CONV_LAYER__

class FWSConvLayer : public Layer {

private:
	int _filter_width;
	int _filter_height;
	int _filter_size;

	int _actv_type;


	double* _u_conv;
	double* _u_dconv;

	double* _u_convb;
	double* _u_dconvb;

	double *_u_vel;
	double *_u_velb;


public:
	FWSConvLayer(Layer *prev=NULL) : Layer(prev, NULL) {
		_filter_size = 0;
		_filter_width = 0;
		_filter_height = 0;
		_u_convb = NULL;
		_u_conv = NULL;
		_u_dconv = NULL;
		_u_dconvb = NULL;
		_u_vel = NULL;
		_u_velb = NULL;
		_actv_type = SIGMOID;
		_layer_type = FWS_CONV_LAYER;
	}

	FWSConvLayer(int fw, int fh, int nm, int at, Layer *prev, Layer *next = NULL) : 	Layer(prev, next) {

		this->_filter_width = fw;
		this->_filter_height = fh;
		this->_filter_size = fw * fh;
		this->_map_num = nm;// * prev->getMapNum();

		int w = prev ? 1 + (prev->getWidth() - fw) : 0;
		int h = prev ? 1 + (prev->getHeight() - fh) : 0;

		if(w < 0 || h < 0) {
			w = 0;
			h = 0;
		}


		this->_unit_count = w * h;
		this->_width = w;
		this->_height = h;
		this->_prev_unit_count = prev ? prev->getUnitCount() : 0;

		this->_actv_type = at;
		this->_layer_type = FWS_CONV_LAYER;

		//_u_dW = NULL;
		_u_conv = NULL;
		_u_dconv = NULL;
		_u_convb = NULL;
		_u_dconvb = NULL;
		_u_vel = NULL;
		_u_velb = NULL;

	}
	~FWSConvLayer() {

		if(_u_conv)
			delete [] _u_conv;
		if(_u_dconv)
			delete [] _u_dconv;
		if(_u_convb)
			delete [] _u_convb;
		if(_u_dconvb)
			delete [] _u_dconvb;
		if(_u_vel)
			delete [] _u_vel;
		if(_u_velb)
			delete [] _u_velb;
	}

	double *getDConv() {
		return _u_dconv;
	}
	double *getConv() {
		return _u_conv;
	}
	double *getConvb() {
		return _u_convb;
	}

	void init() {

		clear();
		double rg = sqrt(6) / sqrt(_unit_count + _prev_unit_count);

		int n = _unit_count, np = _prev_unit_count, nf = _filter_size, nm = _map_num;

		_u_a = new double[n*nm];
		memset(_u_a, 0, n*nm*sizeof(double));

		_u_delta = new double[n*nm];
		memset(_u_delta, 0, n*nm*sizeof(double));


		//conv
		_u_conv = new double[nf*nm];
		for(int i=0;i<nf*nm;i++) {
			_u_conv[i] = ((double) rand() / (RAND_MAX))*2*rg - rg;
		}
		_u_dconv = new double[nf*nm];
		memset(_u_dconv, 0, nf*nm*sizeof(double));

		_u_vel = new double[nf*nm];
		memset(_u_vel, 0, nf*nm*sizeof(double));

		//bias
		_u_convb = new double[nm];
		for(int i=0;i<nm;i++) {
			_u_convb[i] = ((double) rand() / (RAND_MAX))*2*rg - rg;
		}
		_u_dconvb = new double[nm];
		memset(_u_dconvb, 0, nm*sizeof(double));

		_u_velb = new double[nm];
		memset(_u_velb, 0, nm*sizeof(double));

		this->_act_f = Activation::getActivation(_actv_type);
		this->_d_act_f = Activation::getDActivation(_actv_type);

	}

	void forward() {

		//_u_a = _u_W * _prev->getActivation() + _u_convb;// [n, np]*[np, 1]
		int n = _unit_count, np = _prev_unit_count, nm = _map_num, nf = _filter_size;
		int nmp =  _prev->getMapNum();
		int pw = _prev->getWidth();
		int fw = _filter_width;
		int fh = _filter_height;

		double *ua = _u_a;
		double *pua = _prev->getActivation();
		double *cv = _u_conv;
		for(int mi = 0; mi < nm; mi++, ua += n, cv += nf) {

			for(int i = 0; i < n; i++)
				*(ua + i) = _u_convb[mi];

			for(int sh = 0; sh < np * nmp; sh += np) {
				int step = 0, h = 0;
				for(int i = 0; i < n; i++, h++, step++ ) {
					if(h >= _width) {
						step = (step / pw + 1) * pw;
						h = 0;
					}
					double d = 0;
					for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
						for(int k = 0; k < fw; k++ ) {
							d += (*(cv + j2 + k)) * (*(pua + sh + (step + j1) + k));
						}
					}
					*(ua + i) += d;
				}

			}

			_act_f(ua, n);
		}
	}
	void backpropagation() {

		//accumulate dW
		//dW = _u_delta * _prev->getActivation().transpose();
		//[n, 1] *[1, np]
		int n = _unit_count, np = _prev_unit_count, nf = _filter_size, nm = _map_num;
		//int cc = nm / _prev->getMapNum();
		int nmp = _prev->getMapNum();
		int pw = _prev->getWidth();
		int fw = _filter_width;
		int fh = _filter_height;

		//cblas_dscal(nm*nf, mu, _u_dconv, 1);
		// for(int i = 0; i < nm*nf; i++) {
		// 	_u_dconv[i] *= mu;
		// }

		double *dc = _u_dconv;

		double *pua = _prev->getActivation();
		double *dt = _u_delta;

		//feature map loop
		for(int mi = 0; mi < nm; mi++, dt += n, dc += nf) {

			for(int sh = 0; sh < np * nmp; sh += np) {
				int step = 0, h = 0;
				for(int i = 0; i < n; i++, h++, step++ ) {
					if(h >= _width) {
						step = (step / pw + 1) * pw;
						h = 0;
					}
					double d = *(dt + i);
					for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
						for(int k = 0; k < fw; k++ ) {
							*(dc + j2 + k) += d * (*(pua + sh + (step + j1) + k));
						}
					}
				}
			}
		}

		//_u_dconvb = mu*_u_dconvb + _u_delta;
		dt = _u_delta;
		for(int mi = 0; mi < nm; mi++, dt += n) {
			double sum = 0;
			for(int i=0;i<n;i++) {
				sum += dt[i];
			}
			// _u_dconvb[mi] *= mu;
			_u_dconvb[mi] += sum;
		}

		//t = (_u_W.transpose() * _u_delta); // [np, n] * [n, 1]
		//_prev->updateDelta();
		double *pdt = _prev->getDelta();
		if(pdt) {

			memset(pdt, 0, sizeof(double)*_prev->getTotalUnitCount());
			dt = _u_delta;
			double *cv = _u_conv;
			for(int mi = 0; mi < nm; mi++, dt += n, cv += nf) {

				//int sh = np * (mi / cc);
				for(int sh = 0; sh < np * nmp; sh += np) {
					int step = 0, h = 0;
					for(int i = 0; i < n; i++, h++, step++ ) {
						if(h >= _width) {
							step = (step / pw + 1) * pw;
							h = 0;
						}
						double d = *(dt + i);
						for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
							for(int k = 0; k < fw; k++ ) {
								*(pdt + sh + (step + j1) + k) += *(cv + j2 + k) * d;
							}
						}
					}
				}
			}
			_prev->updateDelta();
		}
	}

	void updateDelta() {

		int n = _unit_count, nm = _map_num;
		// mat = ( W'delta )
		// f'(z), where a = f(z) is sigmoid funtion
		_d_act_f(_u_a, n*nm);
		//_u_delta = mat.cwiseProduct(df);
		for(int i = 0; i < n*nm; i++) {
			_u_delta[i] *= _u_a[i];
		}
	}

	void updateParameters(int m, double alpha, double lambda, double mu) {

		int n = _unit_count, nf = _filter_size, nm = _map_num;
		double rm = 1.0 / m;
		//_u_conv = _u_conv - alpha * ( rm * _u_dconv + lambda * _u_conv );
		//cblas_dscal(nf*nm, 1-alpha*lambda, _u_conv, 1);
		//cblas_daxpy(nf*nm, -alpha*rm, _u_dconv, 1, _u_conv, 1);

		for(int i = 0; i < nf*nm; i++) {
			_u_vel[i] = _u_vel[i]*mu + alpha * ( rm * _u_dconv[i] + lambda * _u_conv[i] );
			_u_conv[i] -= _u_vel[i];
		}

		//_u_convb = _u_convb - alpha * (rm * _u_dconvb );
		//cblas_daxpy(nm, -alpha*rm, _u_dconvb, 1, _u_convb, 1);
		for(int i = 0; i < nm; i++) {
			_u_velb[i] = _u_velb[i]*mu + alpha * ( rm * _u_dconvb[i] );
			_u_convb[i] -= _u_velb[i];
		}

		for(int i = 0; i < nm*nf; i++) {
			_u_dconv[i] = 0;
		}
		for(int i = 0; i < nm; i++) {
			_u_dconvb[i] = 0;
		}
	}



	int getTotalUnitCount() {
		return _unit_count * _map_num;
	}

	void clear() {
		Layer::clear();
		if(_u_conv) {
			delete [] _u_conv;
			_u_conv = NULL;
		}
		if(_u_convb) {
			delete [] _u_convb;
			_u_convb = NULL;
		}
		if(_u_dconv) {
			delete [] _u_dconv;
			_u_dconv = NULL;
		}
		if(_u_dconvb) {
			delete [] _u_dconvb;
			_u_dconvb = NULL;
		}
		if(_u_vel) {
			delete [] _u_vel;
			_u_vel = NULL;
		}
		if(_u_velb) {
			delete [] _u_velb;
			_u_velb = NULL;
		}
	}

	void write(std::ofstream &fout) {

		fout << _layer_type << std::endl;
		fout<< _actv_type << " ";
		fout << _unit_count << " " << _prev_unit_count << " ";
		fout << _filter_width << " " << _filter_height << " ";
		fout << _width << " " << _height << " " << _map_num << std::endl;
		for(int i=0;i<_filter_size*_map_num;i++) {
			fout << _u_conv[i] << " ";
		}
		fout<<std::endl;
		for(int i=0;i<_map_num;i++) {
			fout << _u_convb[i] << " ";
		}
		fout<<std::endl;
	}
	void read(std::ifstream &fin) {

		fin >> _actv_type;
		fin >> _unit_count >> _prev_unit_count;
		fin >> _filter_width >> _filter_height;
		fin >> _width >> _height >> _map_num;

		_filter_size = _filter_width * _filter_height;

		init();

		for(int i=0;i<_filter_size*_map_num;i++) {
			fin >> _u_conv[i];
		}
		for(int i=0;i<_map_num;i++) {
			fin >> _u_convb[i];
		}

	}

};


#endif

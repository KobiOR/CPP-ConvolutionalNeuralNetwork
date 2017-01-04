

#include "Layer.hpp"


#ifndef __NN_PWS_CONV_LAYER__

#define __NN_PWS_CONV_LAYER__

class PWSConvLayer : public Layer {

private:
	int _filter_width;
	int _filter_height;
	int _filter_size;
	int _section_width;
	int _section_height;
	int _section_rows;
	int _section_cols;
	int _actv_type;

	//you can use every _stride_x units for computing
	int _stride_x;
	int _stride_y;


	double* _u_conv;
	double* _u_dconv;

	double* _u_convb;
	double* _u_dconvb;

	double* _u_vel;
	double* _u_velb;


public:
	PWSConvLayer(Layer *prev=NULL) : Layer(prev, NULL) {
		_actv_type = SIGMOID;
		_filter_size = 0;
		_filter_width = 0;
		_filter_height = 0;
		_section_width = 0;
		_section_height = 0;
		_section_rows = 0;
		_section_cols = 0;
		_u_convb = NULL;
		_u_conv = NULL;
		_u_dconv = NULL;
		_u_dconvb = NULL;
		_u_vel = NULL;
		_u_velb = NULL;
	}

	PWSConvLayer(int fw, int fh, int sw, int sh, int nm, int stpx, int stpy, int at, Layer *prev, Layer *next = NULL) : 	Layer(prev, next) {

		this->_filter_width = fw;
		this->_filter_height = fh;

		this->_filter_size = fw * fh;
		this->_map_num = nm;// * prev->getMapNum();
		
		this->_section_width = sw;
		this->_section_height = sh;
		this->_stride_x = stpx;
		this->_stride_y = stpy;

		int w = prev ? 1 + (prev->getWidth() - fw) / stpx : 0;
		int h = prev ? 1 + (prev->getHeight() - fh) / stpy : 0;


		if(w < 0 || h < 0) {
			w = 0;
			h = 0;
		}

		this->_section_cols = ceil(double(w)/sw);
		this->_section_rows = ceil(double(h)/sh);

		this->_unit_count = w * h;
		this->_width = w;
		this->_height = h;
		this->_prev_unit_count = prev ? prev->getUnitCount() : 0;
		this->_actv_type = at;
		this->_layer_type = PWS_CONV_LAYER;


		//_u_dW = NULL;
		_u_conv = NULL;
		_u_dconv = NULL;
		_u_convb = NULL;
		_u_dconvb = NULL;
		_u_vel = NULL;
		_u_velb = NULL;

	}
	~PWSConvLayer() {

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
	double *getDConvb() {
		return _u_dconvb;
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


		int ns = _section_rows * _section_cols;
		//conv
		_u_conv = new double[nf*nm*ns];
		for(int i=0;i<nf*nm*ns;i++) {
			_u_conv[i] = ((double) rand() / (RAND_MAX))*2*rg - rg;
		}
		_u_dconv = new double[nf*nm*ns];
		memset(_u_dconv, 0, nf*nm*ns*sizeof(double));

		_u_vel = new double[nf*nm*ns];
		memset(_u_vel, 0, nf*nm*ns*sizeof(double));


		//bias
		_u_convb = new double[nm*ns];
		for(int i=0;i<nm*ns;i++) {
			_u_convb[i] = ((double) rand() / (RAND_MAX))*2*rg - rg;
		}
		_u_dconvb = new double[nm*ns];
		memset(_u_dconvb, 0, nm*ns*sizeof(double));

		_u_velb = new double[nm*ns];
		memset(_u_velb, 0, nm*ns*sizeof(double));

		this->_act_f = Activation::getActivation(_actv_type);
		this->_d_act_f = Activation::getDActivation(_actv_type);

	}

	inline int getSection(int y, int x) {
		return y/_section_height*_section_cols + x/_section_width;
	}

	void forward() {

		//_u_a = _u_W * _prev->getActivation() + _u_convb;// [n, np]*[np, 1]
		int n = _unit_count, np = _prev_unit_count, nm = _map_num, nf = _filter_size;
		int nmp =  _prev->getMapNum();
		int pw = _prev->getWidth();
		int fw = _filter_width;
		int fh = _filter_height;

		//number of sections of a feature map
		int ns = _section_rows * _section_cols;

		double *ua = _u_a;
		double *pua = _prev->getActivation();
		double *cv = _u_conv;
		for(int mi = 0; mi < nm; mi++, ua += n, cv += nf*ns) {

			int x = 0, y = 0;
			for(int i = 0; i < n; i++, x++) {
				if(x >= _width) {
					x = 0; y++;
				}
				*(ua + i) = _u_convb[mi*ns + getSection(y, x)];
			}

			// puast - start position of feature map of pua
			for(int puast = 0; puast < np * nmp; puast += np) {

				int x = 0, y = 0;
				int step = 0;

				// x and y controls step
				for(int i = 0; i < n; i++, x++, step += _stride_x) {
					if(x >= _width) {
						step = (step / pw + _stride_y) * pw;
						x = 0; y++;
					}
					int sec = getSection( y, x );
					//printf("%d %d %d\n", y, x, sec);
					double d = 0;

					//j1 is vertical shift of pua
					//j2 is vertical shift of conv filter
					for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
						for(int k = 0; k < fw; k++ ) {

						//	printf("%lf\n", (*(cv + sec*nf + j2 + k)) * (*(pua + puast + (step + j1) + k)));
							d += (*(cv + sec*nf + j2 + k)) * (*(pua + puast + (step + j1) + k));
						}
					}
					//printf("sum: %lf\n", *(ua+i));
					*(ua + i) += d;
					//printf("sum: %lf\n", *(ua+i));
				}
				//printf("\n");
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

		//number of sections of a feature map
		int ns = _section_rows * _section_cols;

		double *dc = _u_dconv;

		double *pua = _prev->getActivation();
		double *dt = _u_delta;
		for(int mi = 0; mi < nm; mi++, dt += n, dc += nf*ns) {

			//int puast = np * (mi / cc);
			for(int puast = 0; puast < np * nmp; puast += np) {
				int step = 0, x = 0, y = 0;
				for(int i = 0; i < n; i++, x++, step += _stride_x ) {
					if(x >= _width) {
						step = (step / pw + _stride_y) * pw;
						x = 0; y++;
					}
					int sec = getSection( y, x );
					double d = *(dt + i);
					//printf("\n%d %d %d %lf\n\n", y, x, sec, d);
					for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
						for(int k = 0; k < fw; k++ ) {
							*(dc + sec*nf + j2 + k) += d * (*(pua + puast + (step + j1) + k));
						//	printf("%d %lf\n", sec*nf + j2 + k, (*(pua + puast + (step + j1) + k)));
						}
					}
				}
			}
		}

		dt = _u_delta;
		double *dcb = _u_dconvb;
		for(int mi = 0; mi < nm; mi++, dt += n, dcb += ns) {
			int x = 0, y = 0;
			for(int i = 0; i < n; i++, x++ ) {
				if(x >= _width) {
					x = 0; y++;
				}
				int sec = getSection(y,x);
				*(dcb + sec) += *(dt + i);
			}
		}

		//t = (_u_W.transpose() * _u_delta); // [np, n] * [n, 1]
		//_prev->updateDelta();
		double *pdt = _prev->getDelta();
		if(pdt) {

			memset(pdt, 0, sizeof(double)*_prev->getTotalUnitCount());
			dt = _u_delta;
			double *cv = _u_conv;
			for(int mi = 0; mi < nm; mi++, dt += n, cv += nf*ns) {

				//int puast = np * (mi / cc);
				for(int puast = 0; puast < np * nmp; puast += np) {
					int step = 0, x = 0, y = 0;
					for(int i = 0; i < n; i++, x++, step += _stride_x ) {
						if(x >= _width) {
							step = (step / pw + _stride_y) * pw;
							x = 0; y++;
						}
						int sec = getSection( y, x );
						double d = *(dt + i);
						for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
							for(int k = 0; k < fw; k++ ) {
								*(pdt + puast + (step + j1) + k) += *(cv + sec*nf + j2 + k) * d;
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

		int n = _unit_count, nf = _filter_size, nm = _map_num, ns = _section_cols * _section_rows;
		double rm = 1.0 / m;

		//_u_conv = _u_conv - alpha * ( rm * _u_dconv + lambda * _u_conv );
		for(int i = 0; i < nf*nm*ns; i++) {
			//_u_conv[i] *= mu;
			_u_vel[i] = _u_vel[i] * mu + alpha * ( rm * _u_dconv[i] + lambda * _u_conv[i] );
			_u_conv[i] -= _u_vel[i];
			//_u_conv[i] -= alpha * ( rm * _u_dconv[i] + lambda * _u_conv[i] );
		}

		//_u_convb = _u_convb - alpha * (rm * _u_dconvb );
		for(int i = 0; i < nm*ns; i++) {
			//_u_convb[i] *= mu;
			_u_velb[i] = _u_velb[i] * mu + alpha * ( rm * _u_dconvb[i] );
			_u_convb[i] -= _u_velb[i];
			//_u_convb[i] -= alpha * ( rm * _u_dconvb[i] );
		}


		for(int i = 0; i < nm*nf*ns; i++) {
			_u_dconv[i] = 0;
		}
		for(int i = 0; i < nm*ns; i++) {
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
		fout << _actv_type << " ";
		fout << _unit_count << " " << _prev_unit_count << " ";
		fout << _filter_width << " " << _filter_height << " ";
		fout << _section_width << " " << _section_height << " ";
		fout << _section_rows << " " << _section_cols << " ";
		fout << _width << " " << _height << " " << _map_num << std::endl;

		int ns = _section_rows * _section_cols;
		for(int i=0;i<_filter_size*ns*_map_num;i++) {
			fout << _u_conv[i] << " ";
		}
		fout<<std::endl;
		for(int i=0;i<_map_num*ns;i++) {
			fout << _u_convb[i] << " ";
		}
		fout<<std::endl;
	}
	void read(std::ifstream &fin) {

		fin >> _actv_type;
		fin >> _unit_count >> _prev_unit_count;
		fin >> _filter_width >> _filter_height;
		fin >> _section_width >> _section_height;
		fin >> _section_rows >> _section_cols;
		fin >> _width >> _height >> _map_num;


		_filter_size = _filter_width * _filter_height;

		int ns = _section_rows * _section_cols;

		init();

		for(int i=0;i<_filter_size*_map_num*ns;i++) {
			fin >> _u_conv[i];
		}
		for(int i=0;i<_map_num*ns;i++) {
			fin >> _u_convb[i];
		}

	}

};


#endif

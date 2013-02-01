#ifndef __KDES__
#define __KDES__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <set>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "matio.h"
#include "pbm.h"
//#include "liblinear/linear.h"

//#define PI 3.14159265
#define PI M_PI

typedef std::pair<int,int> myindex;
typedef std::pair<float,myindex> mypair;
static bool comparator ( const mypair& l, const mypair& r){ return l.first > r.first; }
static inline bool isnan(float x) { return !(x==x); }

class KernelDescManager
{
 public:
  KernelDescManager();
  ~KernelDescManager();
  
  /*
   * Descriptor Function
   */
  void GKDESDense(MatrixXf& feaArr, MatrixXf& feaMag, MatrixXf& fgrid_y, MatrixXf& fgrid_x, IplImage* im, _kdesStructure& kdes_param, int grid_space, int patch_size, float low_contrast);
  void RGBKDESDense(MatrixXf& feaArr, MatrixXf& feaMag, MatrixXf& fgrid_y, MatrixXf& fgrid_x,
		    IplImage* im, _kdesStructure& kdes_param, int grid_space, int patch_size);
  void SpinKDESDense(MatrixXf& feaArr, MatrixXf& fgrid_y, MatrixXf& fgrid_x, IplImage* im, const MatrixXf& top_left,
		     _kdesStructure& kdes_param, int grid_space, int patch_size, double normal_thresh=0.01, double normal_window=5);
  //void SpinKDESDense(MatrixXf& feaArr, MatrixXf& fgrid_y, MatrixXf& fgrid_x, IplImage* im, const MatrixXf& top_left, MatIO::matvarplus_t* kdes_params, int grid_space, int patch_size, double normal_thresh=0.01, double normal_window=5);
  void CKSVDEMK(MatrixXf& imfea, const MatrixXf& feaArr, const MatrixXf& feaMag, const MatrixXf& fgrid_y, const MatrixXf& fgrid_x, const int img_h, const int img_w, MatrixXf& words, MatrixXf& G, MatrixXf& pyramid, const float kparam);
  
  /*
   * Eval Kernel Function
   */
  MatrixXf EvalKernelExp( const MatrixXf& data1, const MatrixXf& data2, float kparam );
  MatrixXf EvalKernelExp_d( const MatrixXf& data1_f, const MatrixXf& data2_f, float kparam );
  MatrixXf EvalKernelExp_Img2(const MatrixXf& I_ox, const MatrixXf& I_oy,
			      const MatrixXf& sample2, MatrixXf& params);
  MatrixXf EvalKernelExp_Img3(const MatrixXf& im1, const MatrixXf& im2, const MatrixXf& im3,
			      const MatrixXf& sample2, MatrixXf& params);
  
  /*
   * Others
   */
  IplImage* rgb2gray(const IplImage* im);
  void isotropic_gaussian(MatrixXf&  result, float sigma, int  patch_size);
  void depth2cloud(const MatrixXf& depth, MatrixXf& pcloud_x, MatrixXf& pcloud_y, MatrixXf& pcloud_z,
		   const MatrixXf& top_left, double focal=570.3, double ycenter=240.0, double xcenter=320.0);
  void pcnormal(const MatrixXf& pcloud_x, const MatrixXf& pcloud_y, const MatrixXf& pcloud_z,
		MatrixXf& normal_x, MatrixXf& normal_y, MatrixXf& normal_z, double normal_thresh, double normal_window, bool pos_z=true );
  MatrixXf scaletest_linear(const MatrixXf& imfea, const MatrixXf& minvalue, const MatrixXf& maxvalue);
  //MatrixXf scaletest_power(const MatrixXf& imfea, const MatrixXf& minvalue, const MatrixXf& maxvalue)
  
  //bool initModel();
  //bool Process(MatrixXf&imfea, IplImage* image);
  //bool Process(MatrixXf&imfea, IplImage* image, const VectorXf& top_left);
  //predict is other class...maybe
  //string GetObjectName(MatrixXf& imfea);
  //string GetObjectName(MatrixXf& imfea, vector<string>& top_objects, vector<double>& top_scores);
  //void PrintModelList();
  
 private:
  
};
#endif

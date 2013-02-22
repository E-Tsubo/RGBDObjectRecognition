#ifndef __PBM_
#define __PBM_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "matio.h"
#include "benchmark-utils.hpp"
#include "liblinear/linear.h"

const std::string MODEL_HEAD = "/MODEL.txt";
const std::string PART_DETECTOR_PATH = "/PD";
const std::string JOINT_DETECTOR_PATH = "/JD";
const std::string MODEL_EXTENSION = ".model";

#define MULTITHREAD_PART 1
#define MULTITHREAD_FEA 0

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class KernelDescManager;

typedef struct{
  float grid_space;
  float patch_size;
  float low_contrast;//only gradkdes
  
  //kdes
  MatrixXf gpoints;//rgb, g, n points
  MatrixXf spoints, kparam;
  MatrixXf kparam_eigen;
  
  //emk
  MatrixXf emkWords, emkG, emkPyramid;
  float emkKparam;
} _kdesStructure;

typedef struct{
  std::string part;
  int feaNum;
  std::vector<std::string> kdesName;
  
  std::vector<MatIO::matvarplus_t*> kdes;
  std::vector<_kdesStructure> kdesparam;
  struct model* svmModel;
  std::string svmType;
  std::string normalType;
  MatrixXf minvalue, maxvalue;
} _pDetector;

typedef struct{
  std::string modelName;
  std::string svmDatName;
  std::string classNameFile;
  
  MatIO::matvarplus_t* svmData;
  struct model* svmModel;
  //struct svm_model _svm;
  std::string svmType;
  std::string normalType;
  MatrixXf minvalue, maxvalue;
  
  std::vector<std::string> className;
} _jDetector;

typedef struct{
  std::vector<double> jd_dec_values;
  std::vector<std::size_t> idx;
  //double* jd_dec_values;
  int jd_predictlabel;
  //int pd_predictlabel;
} _predictResult;

class compare_descending
{
private:
    std::vector<double> const& m_v;

public:
    typedef bool result_type;
    
    compare_descending(std::vector<double> const& _v)
        : m_v(_v)
    {}
    
    bool operator()(std::size_t a, std::size_t b) const
    {
        return (m_v[a] > m_v[b]);
    }
};

class PBM
{
 public:
  PBM(char*);
  ~PBM();
  bool readModelHead( std::string );
  bool setData();
  bool setData_PD();
  bool setData_JD();
  bool loadKdesParam();
    
  //bool loadKdes_();
  MatIO::matvarplus_t* loadKdes_mat( const char* program_name, const char* path );
  void loadClassName(std::string path);
  
  double Process(KernelDescManager& kdm, IplImage* rgb, IplImage* dep, MatrixXf& top_left);
  void pDetectorProcess(KernelDescManager& kdm, IplImage* rgb, IplImage* dep, MatrixXf& top_left, double** dec);
  double jDetectorProcess(KernelDescManager& kdm, double** dec);
  void pDetectorProcessFea(MatrixXf* imfea, KernelDescManager& kdm,
			   IplImage* rgb, IplImage* dep, MatrixXf& top_left);
  bool extractFea( int feaIdx, MatrixXf& feaArr, MatrixXf& feaMag,
		   MatrixXf& fgrid_x, MatrixXf& fgrid_y,
		   KernelDescManager& kdm,
		   IplImage* rgb, IplImage* dep, MatrixXf& top_left );
  void getScore(KernelDescManager& kdm, MatrixXf* imfea, double** dec);
  void getScore_liblinear(KernelDescManager& kdm, MatrixXf& imfea, double* dec, int pdIdx);
  //void getScore_libsvm(MatrixXf* imfea, double* dec);
  
  std::string getObjName( int idx )
    {
      return m_jDetector->className[idx];
    };
  
  void setResult( int label, double* dec_values )
  {
    m_predictData->jd_dec_values.clear();
    m_predictData->jd_predictlabel = label;
    for( int i = 0; i < m_jDetector->className.size(); i++ )
      m_predictData->jd_dec_values.push_back( dec_values[i] );
    /*
    std::sort( m_predictData->jd_dec_values.begin(),
	       m_predictData->jd_dec_values.end(),
	       greater<double>() );
    */
    m_predictData->idx.resize( m_jDetector->className.size() );
    for( int i = 0; i < m_predictData->idx.size(); i++ )
      m_predictData->idx[i] = i;
    
    std::sort( m_predictData->idx.begin(),
	       m_predictData->idx.end(),
	       compare_descending(m_predictData->jd_dec_values) );
    for(int i = 0; i < m_predictData->idx.size(); i++ )
      std::cerr << m_predictData->idx[i] << " "
		<< m_predictData->jd_dec_values[m_predictData->idx[i]] << std::endl;
  };
  
 private:
  std::string m_modelPath;
    
  //PBM Configuration Var//
  int m_pdNum;
  _pDetector *m_pDetector;
  _jDetector *m_jDetector;
  std::vector<std::string> m_className;
  
  //Result Var//
  _predictResult *m_predictData;
};
#endif

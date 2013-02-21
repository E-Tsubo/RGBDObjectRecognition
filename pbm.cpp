#include "pbm.h"
#include "kerneldesc.h"

using namespace MatIO;

/**
 * @return void
 * @param[in] path Part-based Model's path
 */
PBM::PBM(char* path)
{
  m_modelPath = std::string(path);
  readModelHead( m_modelPath+MODEL_HEAD );
  setData();
}

/**
 * delete Part-based Model Structure
 */
PBM::~PBM()
{
  //delete(pDetector.kdes)TODO!!
  delete [] m_pDetector;
  delete m_jDetector;
}

/**
 * モデル情報を格納したヘッダファイルをロード
 * @return bool success or false
 * @param[in] filePath File path
 */
bool PBM::readModelHead( std::string filePath )
{
  std::ifstream input( filePath.c_str() );
  std::stringstream ss; std::string str, tmp;
  int cnt;
  
  //PARTNUM
  std::getline( input, str );
  ss.str(""); ss.clear(); ss.str(str);
  ss >> tmp >> m_pdNum;
  /* Set up the structure of Part-Detector */
  m_pDetector = new _pDetector[m_pdNum];
  
  //FEATURENUM
  std::getline( input, str );
  ss.str(""); ss.clear(); ss.str(str);
  ss >> tmp;
  cnt = 0;
  while( !ss.eof() ){
    ss >> m_pDetector[cnt].feaNum;
    std::cout << m_pDetector[cnt].feaNum << std::endl;
    cnt++;
  }
  
  //FEATURE
  std::getline( input, str );
  for( int i = 0; i < m_pdNum; i++ ){
    std::getline( input, str );
    ss.str(""); ss.clear(); ss.str(str);
    ss >> m_pDetector[i].part;
    
    while( !ss.eof() ){
      ss >> tmp;
      m_pDetector[i].kdesName.push_back(tmp);
      //std::cout << m_pDetector[i].kdesName[0].c_str() << " ";
    }
    //std::cout << std::endl;
  }
  
  //JOINTMODEL
  m_jDetector = new _jDetector;
  std::getline( input, str );
  ss.str(""); ss.clear(); ss.str(str);
  ss >> tmp >> m_jDetector->modelName;
  
  //MINVAXVALUE
  std::getline( input, str );
  ss.str(""); ss.clear(); ss.str(str);
  ss >> tmp >> m_jDetector->svmDatName;
  
  //CLASSNAME
  std::getline( input, str );
  ss.str(""); ss.clear(); ss.str(str);
  ss >> tmp >> m_jDetector->classNameFile;
  
  input.close();
  return true;
}

/**
 * @return bool success or false
 */
bool PBM::setData()
{
  setData_PD();
  setData_JD();
  loadKdesParam();
  return true;
}

/**
 * Part-Detectorをロード
 * @return bool success or false
 */
bool PBM::setData_PD()
{
  for( int i = 0; i < m_pdNum; i++ ){
    std::string path = m_modelPath + PART_DETECTOR_PATH + std::string("/") + m_pDetector[i].part;
    
    for( int j = 0; j < m_pDetector[i].feaNum; j++ ){
      
      /****************************************************************************************
       * kerneldescriptor
       ***************************************************************************************/
      std::string filepath = path + std::string("/") + m_pDetector[i].kdesName[j];
      std::cout << "Loading... " << filepath << std::endl;
      
      //loadKdesParam( m_pDetector[i],  );setup to var
      matvarplus_t* tmp = loadKdes_mat( "Param file", filepath.c_str());
      m_pDetector[i].kdes.push_back(tmp);
      //debug
      //MatrixXf gpoints;
      //get_matrix(gpoints, m_pDetector[i].kdes[j], "test->svm->w");
    }
    
    //Joint FEATURE Single FeatureでなくJoint-Featureであるならばjointkdes.matをロード
    if( m_pDetector[i].feaNum != 1 ){
      std::string filepath = path + std::string("/") + m_pDetector[i].kdesName[m_pDetector[i].feaNum];
      std::cout << "Loading... " << filepath << std::endl;
      matvarplus_t* tmp = loadKdes_mat( "Param file", filepath.c_str());
      m_pDetector[i].kdes.push_back(tmp);
    }
    
    /****************************************************************************************
     * svm(liblinear) Part-Detectorはliblinearのみをサポート
     ***************************************************************************************/
    std::string filepath = path + std::string("/") + m_pDetector[i].part + std::string(MODEL_EXTENSION);
    std::cout << "Loading... " << filepath << std::endl;
    if( (m_pDetector[i].svmModel = load_model( filepath.c_str() ) ) == 0 ){
      std::cerr << "setData_PD():can't open part-detector model" << std::endl;
      return false;
    }
    
  }
  return true;
}

/**
 * Joint-Detectorをロード
 * @return bool success or false
 */
bool PBM::setData_JD()
{
  std::string filepath = m_modelPath + JOINT_DETECTOR_PATH + std::string("/") + m_jDetector->svmDatName;
  /****************************************************************************************
   * svm param(for normailzation value)
   ***************************************************************************************/
  std::cout << "Loading... " << filepath << std::endl;
  m_jDetector->svmData = loadKdes_mat( "Param file", filepath.c_str() );
  
  /****************************************************************************************
   * svm(liblinear or libsvm) TODO only support liblinear now
   ***************************************************************************************/
  filepath = m_modelPath + JOINT_DETECTOR_PATH + std::string("/") + m_jDetector->modelName;
  std::cout << "Loading... " << filepath << std::endl;
  if( (m_jDetector->svmModel = load_model( filepath.c_str() ) ) == 0 ){
    std::cerr << "setData_JD():can't open joint-detector model" << std::endl;
    return false;
  }
  //minvalue, maxvalue
  get_matrix(m_jDetector->minvalue, m_jDetector->svmData,"svmdat->minvalue"); 
  get_matrix(m_jDetector->maxvalue, m_jDetector->svmData,"svmdat->maxvalue"); 
  
  /****************************************************************************************
   * svm param(classname)
   ***************************************************************************************/
  filepath = m_modelPath + JOINT_DETECTOR_PATH + std::string("/") + m_jDetector->classNameFile;
  loadClassName(filepath);
  
  return true;
}

/**
 * 特徴抽出に用いるパラメータ(Kdes Param)をロード. Matlabバイナリからロードする際にMatIOがスレッドセーフでないので、事前展開をここで行う.
 * @return bool success or false
 */
bool PBM::loadKdesParam()
{
  std::cerr << "setup kdes param from .mat file" << std::endl;
  for( int i = 0; i < m_pdNum; i++ ){
    for( int j = 0; j < m_pDetector[i].feaNum; j++ ){
      
      _kdesStructure tmp;
      //kdes
      if( m_pDetector[i].kdesName[j] == std::string("rgbkdes.mat") )
	{
	  tmp.grid_space=get_value<float>(m_pDetector[i].kdes[j], "rgbkdes->kdes->grid_space");
	  tmp.patch_size=get_value<float>(m_pDetector[i].kdes[j], "rgbkdes->kdes->patch_size");
	  
	  get_matrix(tmp.gpoints, m_pDetector[i].kdes[j], "rgbkdes->kdes->kdes_params->rgbpoints");
	  get_matrix(tmp.spoints, m_pDetector[i].kdes[j], "rgbkdes->kdes->kdes_params->spoints");
	  get_matrix(tmp.kparam, m_pDetector[i].kdes[j], "rgbkdes->kdes->kdes_params->kparam");
	  get_matrix(tmp.kparam_eigen, m_pDetector[i].kdes[j], "rgbkdes->kdes->kdes_params->eigvectors");
	  
	  get_matrix(tmp.emkWords, m_pDetector[i].kdes[j], "rgbkdes->emk->words");
	  get_matrix(tmp.emkG, m_pDetector[i].kdes[j], "rgbkdes->emk->G");
	  get_matrix(tmp.emkPyramid, m_pDetector[i].kdes[j], "rgbkdes->emk->pyramid");
	  tmp.emkKparam=get_value<float>(m_pDetector[i].kdes[j], "rgbkdes->emk->kparam");
	}
      if( m_pDetector[i].kdesName[j] == std::string("gradkdes.mat") )
	{
	  tmp.grid_space=get_value<float>(m_pDetector[i].kdes[j], "gradkdes->kdes->grid_space");
	  tmp.patch_size=get_value<float>(m_pDetector[i].kdes[j], "gradkdes->kdes->patch_size");
	  tmp.low_contrast = 0.8;
	  
	  get_matrix(tmp.gpoints, m_pDetector[i].kdes[j], "gradkdes->kdes->kdes_params->gpoints");
	  get_matrix(tmp.spoints, m_pDetector[i].kdes[j], "gradkdes->kdes->kdes_params->spoints");
	  get_matrix(tmp.kparam, m_pDetector[i].kdes[j], "gradkdes->kdes->kdes_params->kparam");
	  get_matrix(tmp.kparam_eigen, m_pDetector[i].kdes[j], "gradkdes->kdes->kdes_params->eigvectors");

	  get_matrix(tmp.emkWords, m_pDetector[i].kdes[j], "gradkdes->emk->words");
	  get_matrix(tmp.emkG, m_pDetector[i].kdes[j], "gradkdes->emk->G");
	  get_matrix(tmp.emkPyramid, m_pDetector[i].kdes[j], "gradkdes->emk->pyramid");
	  tmp.emkKparam=get_value<float>(m_pDetector[i].kdes[j], "gradkdes->emk->kparam");
	}
      if( m_pDetector[i].kdesName[j] == std::string("gradkdes_dep.mat") )
	{
	  tmp.grid_space=get_value<float>(m_pDetector[i].kdes[j], "gradkdes_dep->kdes->grid_space");
	  tmp.patch_size=get_value<float>(m_pDetector[i].kdes[j], "gradkdes_dep->kdes->patch_size");
	  tmp.low_contrast = 0.8;
	  
	  get_matrix(tmp.gpoints, m_pDetector[i].kdes[j], "gradkdes_dep->kdes->kdes_params->gpoints");
	  get_matrix(tmp.spoints, m_pDetector[i].kdes[j], "gradkdes_dep->kdes->kdes_params->spoints");
	  get_matrix(tmp.kparam, m_pDetector[i].kdes[j], "gradkdes_dep->kdes->kdes_params->kparam");
	  get_matrix(tmp.kparam_eigen, m_pDetector[i].kdes[j], "gradkdes_dep->kdes->kdes_params->eigvectors");

	  get_matrix(tmp.emkWords, m_pDetector[i].kdes[j], "gradkdes_dep->emk->words");
	  get_matrix(tmp.emkG, m_pDetector[i].kdes[j], "gradkdes_dep->emk->G");
	  get_matrix(tmp.emkPyramid, m_pDetector[i].kdes[j], "gradkdes_dep->emk->pyramid");
	  tmp.emkKparam=get_value<float>(m_pDetector[i].kdes[j], "gradkdes_dep->emk->kparam");
	}
      if( m_pDetector[i].kdesName[j] == std::string("normalkdes.mat") )
	{
	  tmp.grid_space=get_value<float>(m_pDetector[i].kdes[j], "normalkdes->kdes->grid_space");
	  tmp.patch_size=get_value<float>(m_pDetector[i].kdes[j], "normalkdes->kdes->patch_size");
	  
	  get_matrix(tmp.gpoints, m_pDetector[i].kdes[j], "normalkdes->kdes->kdes_params->npoints");
	  get_matrix(tmp.spoints, m_pDetector[i].kdes[j], "normalkdes->kdes->kdes_params->spoints");
	  get_matrix(tmp.kparam, m_pDetector[i].kdes[j], "normalkdes->kdes->kdes_params->kparam");
	  get_matrix(tmp.kparam_eigen, m_pDetector[i].kdes[j], "normalkdes->kdes->kdes_params->eigvectors");

	  get_matrix(tmp.emkWords, m_pDetector[i].kdes[j], "normalkdes->emk->words");
	  get_matrix(tmp.emkG, m_pDetector[i].kdes[j], "normalkdes->emk->G");
	  get_matrix(tmp.emkPyramid, m_pDetector[i].kdes[j], "normalkdes->emk->pyramid");
	  tmp.emkKparam=get_value<float>(m_pDetector[i].kdes[j], "normalkdes->emk->kparam");
	}
      
      m_pDetector[i].kdesparam.push_back(tmp);
    }
    
    //minvalue, maxvalue
    if( m_pDetector[i].feaNum != 1 ){
      get_matrix(m_pDetector[i].minvalue, m_pDetector[i].kdes[m_pDetector[i].feaNum],"jointkdes->svm->minvalue"); 
      get_matrix(m_pDetector[i].maxvalue, m_pDetector[i].kdes[m_pDetector[i].feaNum],"jointkdes->svm->maxvalue"); 
    }else{
      int loc = m_pDetector[i].kdesName[0].find(".mat");
      std::string tmp = m_pDetector[i].kdesName[0].erase(loc);
      get_matrix(m_pDetector[i].minvalue, m_pDetector[i].kdes[0], ( tmp+std::string("->svm->minvalue") ).c_str() );
      get_matrix(m_pDetector[i].maxvalue, m_pDetector[i].kdes[0], ( tmp+std::string("->svm->maxvalue") ).c_str() );
      //このコードが動作すれば上記の冗長なコードを簡略化できる
    }
  }

}

/**
 *
 * @return matvarplus_t* matio structure pointer
 * @param[in] program_name const char* 
 * @param[in] mat_file File Path
 */
matvarplus_t* PBM::loadKdes_mat(const char* program_name,const char* mat_file)
{
  mat_t* mat = NULL;
  matvar_t* matvar = NULL;
  Mat_LogInit(program_name);
  mat = Mat_Open(mat_file, MAT_ACC_RDONLY);
  if (!mat){
    Mat_Error("Error opening %s\n", mat_file);
    return NULL;//add
  }
  while ((matvar = Mat_VarReadNext(mat)) != NULL) {
    matvarplus_t* temp =  new matvarplus_t(matvar,NULL); // doesn't destruct in memory, so TODO
    print_default(temp);
    Mat_Close(mat);
    return temp;
  }
}

/**
 * @return void
 * @loadClassName[in] path File Path about Trained class name.
 */
void PBM::loadClassName(std::string path)
{
  std::ifstream input( path.c_str() );
  std::stringstream ss; std::string str, tmp;
  
  std::getline( input, str );
  ss.str(""); ss.clear(); ss.str(str);
  while( !ss.eof() ){
    ss >> tmp;
    m_jDetector->className.push_back(tmp);
    std::cout << tmp << std::endl;
  }
  std::cout << "Training Object Class Num is " << m_jDetector->className.size() << std::endl;
  input.close();
}

/**
 * メインプロセス関数 特徴抽出、認識を行う
 * @return double Prediction label
 * @param[in] kdm this class is charged of extracting features
 * @param[in] rgb RGB Image from kinect
 * @param[in] dep Depth Image from kinect
 * @param[in] top_left Cropped Image Pos 
 */
double PBM::Process(KernelDescManager& kdm, IplImage* rgb, IplImage* dep, MatrixXf& top_left)
{
  double** dec_values = Malloc(double*, m_pdNum);
  for( int i = 0; i < m_pdNum; i++ ){
    dec_values[i] = Malloc(double, m_jDetector->className.size());
  }
  
  pDetectorProcess(kdm, rgb, dep, top_left, dec_values);
  double predict_label = jDetectorProcess(kdm, dec_values );//pDetectorProcess's resutl
  
  for( int i = 0; i < m_pdNum; i++ )
    free( dec_values[i] );
  free( dec_values );
  
  return predict_label;
}

/**
 * Part-Detectorに関する処理を担当,特徴抽出、認識。Joint-Detectorのためにスコアを格納
 * @return void
 * @param[in] kdm this class is charged of extracting features
 * @param[in] rgb RGB Image from kinect
 * @param[in] dep Depth Image from kinect
 * @param[in] top_left Cropped Image Pos 
 * @param[out] dec_values The scores from SVM. This score is the distance from hyper plane.
 */
void PBM::pDetectorProcess(KernelDescManager& kdm, IplImage* rgb, IplImage* dep,
			   MatrixXf& top_left, double** dec_values)
{
  //calc feature
  MatrixXf imfea[m_pdNum];
  pDetectorProcessFea( imfea, kdm, rgb, dep, top_left );
  
  //get score from svm
  getScore( kdm, imfea, dec_values );  
}

/** 
 * Joint-Detectorに関する処理を担当. 
 * @return predict_label
 * @param[in] kdm this class is charged of extracting features
 * @param[in] dec_values the scores from Part-Detector
 */
double PBM::jDetectorProcess(KernelDescManager& kdm, double** dec_values)
{
  //Liblinear only libsvm TODO
  int nr_class = get_nr_class( m_jDetector->svmModel );
  int nr_feature = get_nr_feature( m_jDetector->svmModel );
  
  MatrixXf imfea(nr_feature,1);
  for( int i = 0; i < nr_class; i++ ){//Class label
    for( int j = 0; j < m_pdNum; j++ ){//Part-Detectorごとに
      imfea(i*m_pdNum+j,0) = dec_values[j][i];
    }
  }
  
  MatrixXf minvalue = m_jDetector->minvalue;
  MatrixXf maxvalue = m_jDetector->maxvalue;
  MatrixXf imfea_s = kdm.scaletest_linear( imfea, minvalue, maxvalue);
  
  struct feature_node *x = (struct feature_node *) malloc( (imfea_s.rows()+1) * sizeof(struct feature_node) );
  for( int i = 0; i < imfea_s.rows(); i++ ){
    x[i].index = i+1;
    x[i].value = imfea_s( i, 0 );
  }
  x[imfea_s.rows()].index = -1;
  
  if( m_jDetector->svmModel->bias >= 0 ){
    std::cerr << "model->bias >= 0! Not supported!!" << std::endl;
  }
  
  double* joint_dec_values = Malloc(double, nr_class);
  double predict_label = predict_values( m_jDetector->svmModel, x, joint_dec_values );
  std::cerr << "Joint-Detector Predict Lable " << predict_label << std::endl;//debug
  free(x);
  
  return predict_label;
}

/**
 * 各Part-Detectorように特徴量を抽出. OpenMPによるマルチスレッドにて実行される.最も処理時間が掛かる部分
 * @return void
 * @param[out] imfea(std::vector) 抽出した特徴量を格納
 * @param[in] kdm this class is charged of extracting features
 * @param[in] rgb RGB Image
 * @param[in] dep Depth Image
 * @param[in] top_left Cropped Image Pos
 */
void PBM::pDetectorProcessFea(MatrixXf* imfea, KernelDescManager& kdm,
			      IplImage* rgb, IplImage* dep, MatrixXf& top_left)
{
  Timer timer;
  double exec_time, exec_time2;
  timer.start();
  
  //現状、各part-Detectorの特徴量は同一と仮定している 事実、現状では同一であるので問題ない。
  //この仮定により特徴量の計算を一度で済ませ、計算コストと下げている。
  //なお、Matlabではすべて愚直に計算をしている
  
  MatrixXf tmp_feaArr[m_pDetector[0].feaNum];
  MatrixXf tmp_feaMag[m_pDetector[0].feaNum];
  MatrixXf tmp_fgrid_x[m_pDetector[0].feaNum];
  MatrixXf tmp_fgrid_y[m_pDetector[0].feaNum];
  
  if( m_pDetector[0].feaNum == 1 ){
    //Single Feature
    
  }else{
    //Multi Feature( RGB-D Joint Feature )
    int threadNum = std::max( m_pDetector[0].feaNum, m_pdNum );
#if MULTITHREAD_PART  
    omp_set_num_threads(threadNum);
    std::cerr << "[OpenMP] Enable Thread Num. " << omp_get_max_threads() << std::endl;
#pragma omp parallel
    {
#pragma omp for
#endif
      for( int loopfea = 0; loopfea < m_pDetector[0].feaNum; loopfea++ ){//各特徴量に関して計算
	extractFea( loopfea, tmp_feaArr[loopfea], tmp_feaMag[loopfea], tmp_fgrid_x[loopfea], tmp_fgrid_y[loopfea], kdm, rgb, dep, top_left );
      }
    
#if MULTITHREAD_PART  
#pragma omp for 
#endif
      for( int looppd = 0; looppd < m_pdNum; looppd++ ){//各part-detectorモデル用にbag of feature化
	MatrixXf tmp_imfea[m_pDetector[looppd].feaNum];
	std::cerr << " CVSVDEMK Thread " << looppd+1 << std::endl;//debug
	for( int j = 0; j < m_pDetector[looppd].feaNum; j++ ){
	  kdm.CKSVDEMK( tmp_imfea[j], tmp_feaArr[j], tmp_feaMag[j],
			tmp_fgrid_y[j], tmp_fgrid_x[j], rgb->height, rgb->width,
			m_pDetector[looppd].kdesparam[j].emkWords,
			m_pDetector[looppd].kdesparam[j].emkG,
			m_pDetector[looppd].kdesparam[j].emkPyramid,
			m_pDetector[looppd].kdesparam[j].emkKparam );
	}
	
	//Joint
	int dim = 0;
	for( int j = 0; j < m_pDetector[looppd].feaNum; j++ ){
	  //std::cout << "test"<<k << std::endl;//debug
	  dim += tmp_imfea[j].rows();
	}
	int index = 0;
	(imfea[looppd]).resize(dim,1);
	for( int j = 0; j < m_pDetector[looppd].feaNum; j++ )
	  for( int k = 0; k < tmp_imfea[j].rows(); k++ ){
	    (imfea[looppd])(index,0) = tmp_imfea[j](k,0);
	    index++;
	  }
	
      }
      
    }//OpenMP [omp parallel]
    
  }
  
  exec_time = timer.get();
  cout << "KDES Execution time... " << exec_time << endl;
}

/**
 * 画像より特徴抽出
 * @return bool
 * @param[in] feaIdx feature index No. 
 * @param[out] feaArr 抽出した特徴量を格納
 * @param[out] feaMag
 * @param[out] fgrid_x
 * @param[out] fgrid_y
 * @param[in] kdm KernelDescManager Class
 * @param[in] rgb RGB Image
 * @param[in] dep Depth Image
 * @param[in] top_left cropped image's pos.
 */
bool PBM::extractFea( int feaIdx, MatrixXf& feaArr, MatrixXf& feaMag, MatrixXf& fgrid_x, MatrixXf& fgrid_y,
		      KernelDescManager& kdm,
		      IplImage* _rgb, IplImage* _dep, MatrixXf& _top_left )
{
  IplImage* rgb_init = cvCreateImage(cvSize(_rgb->width, _rgb->height), IPL_DEPTH_32F, _rgb->nChannels);
  IplImage* dep_init = cvCreateImage(cvSize(_dep->width, _dep->height), IPL_DEPTH_32F, _dep->nChannels);
  if (!rgb_init && !dep_init){
    printf("Image is unavailable!\n");
    return false;
  }
  
  /****************************************************************************************
   * Normal
   ***************************************************************************************/
  assert( _dep->nChannels==1 );//must be grayscale(depth image)
  cvConvertScale(_dep, dep_init,1.0/1000, 0);
  cvConvertScale(_rgb, rgb_init,1.0/255, 0);//normalized
  
  /****************************************************************************************
   * Resize
   ***************************************************************************************/
  IplImage* rgb, *dep;
  
  const double EPS_RATIO=0.0001;
  int max_imsize = 300;
  int min_imsize = 45;//TODO
  int MAX_IMAGE_SIZE = 300;
  //int max_imsize=(int)get_value<float>(this->model_kdes, (model_var+"->kdes->max_imsize").c_str() );
  //int min_imsize=(int)get_value<float>(this->model_kdes, (model_var+"->kdes->min_imsize").c_str() );
  
  double ratio, ratio_f, ratio_max, ratio_min;
  ratio_f=1.0;
  if (MAX_IMAGE_SIZE>0) {
    ratio_f = max( ratio_f, max( (double)rgb_init->width/MAX_IMAGE_SIZE, (double)rgb_init->height/MAX_IMAGE_SIZE ) );
  }
  ratio_max = max( max( (double)rgb_init->width/max_imsize, (double)rgb_init->height/max_imsize ), 1.0 );
  ratio_min = min( min( (double)rgb_init->width/min_imsize, (double)rgb_init->height/min_imsize ), 1.0 );
  if (ratio_min<1.0-EPS_RATIO) {
    ratio=ratio_min;
  } else {
    ratio=max(ratio_f,ratio_max);
  }
  
  if (ratio>1.0-EPS_RATIO || ratio<1.0-EPS_RATIO) {
    rgb = cvCreateImage( cvSize((rgb_init->width)/ratio,(rgb_init->height)/ratio), IPL_DEPTH_32F, rgb_init->nChannels);
    dep = cvCreateImage( cvSize((rgb_init->width)/ratio,(rgb_init->height)/ratio), IPL_DEPTH_32F, dep_init->nChannels);
    int method=CV_INTER_CUBIC;
    cvResize( rgb_init, rgb, method );
    method=CV_INTER_NN;   // nearest neighbor for depth image
    cvResize( dep_init, dep, method );
    std::cout << "Resizing..." << std::endl;
  } else {
    rgb = cvCreateImage( cvSize( rgb_init->width, rgb_init->height ), IPL_DEPTH_32F, rgb_init->nChannels );
    dep = cvCreateImage( cvSize( dep_init->width, dep_init->height ), IPL_DEPTH_32F, dep_init->nChannels );
    rgb = rgb_init;
    dep = dep_init;
  }
  
  int rgb_w=rgb->width, rgb_h=rgb->height;
  int dep_w=dep->width, dep_h=dep->height;
  cvReleaseImage(&rgb_init);
  cvReleaseImage(&dep_init);
  
  /****************************************************************************************
   * Extract Featurte
   ***************************************************************************************/
  if( m_pDetector[0].kdesName[feaIdx] == std::string("rgbkdes.mat") )
    {
      std::cerr << "RGB Color Kernel Descriptor" << std::endl;//debug
      kdm.RGBKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, rgb, m_pDetector[0].kdesparam[feaIdx],
		   m_pDetector[0].kdesparam[feaIdx].grid_space,
		   m_pDetector[0].kdesparam[feaIdx].patch_size);
    }
  if( m_pDetector[0].kdesName[feaIdx] == std::string("gradkdes.mat") )
    {
      //Model作成時にlow_contrastを配備すること TODO!!
      std::cerr << "Gradient Kernel Descriptor" << std::endl;//debug
      kdm.GKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, rgb, m_pDetector[0].kdesparam[feaIdx],
		     m_pDetector[0].kdesparam[feaIdx].grid_space,
		     m_pDetector[0].kdesparam[feaIdx].patch_size,
		     m_pDetector[0].kdesparam[feaIdx].low_contrast);
    }
  if( m_pDetector[0].kdesName[feaIdx] == std::string("gradkdes_dep.mat") )
    {
      std::cerr << "Gradient Kernel Descriptor over Depth" << std::endl;//debug
      kdm.GKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, dep, m_pDetector[0].kdesparam[feaIdx],
		     m_pDetector[0].kdesparam[feaIdx].grid_space,
		     m_pDetector[0].kdesparam[feaIdx].patch_size,
		     m_pDetector[0].kdesparam[feaIdx].low_contrast);
    }
  if( m_pDetector[0].kdesName[feaIdx] == std::string("normalkdes.mat") )
    {
      std::cerr << "Spin Image Kernel Descriptor" << std::endl;//debug
      kdm.SpinKDESDense(feaArr, fgrid_y, fgrid_x, dep, _top_left, m_pDetector[0].kdesparam[feaIdx],
			m_pDetector[0].kdesparam[feaIdx].grid_space,
			m_pDetector[0].kdesparam[feaIdx].patch_size);
      /*// old functions
	kdm.SpinKDESDense(feaArr, fgrid_y, fgrid_x, dep, _top_left, m_pDetector[0].kdes[feaIdx],
	get_value<float>(m_pDetector[0].kdes[feaIdx], "normalkdes->kdes->grid_space"),
	get_value<float>(m_pDetector[0].kdes[feaIdx], "normalkdes->kdes->patch_size"));
      */
    }
  
  cvReleaseImage(&rgb);
  cvReleaseImage(&dep);
  //cvReleaseImage(&rgb_init);
  //cvReleaseImage(&dep_init);
}

/**
 * 各Part-Detectorからスコアを取得
 * @return void
 * @param[in] kdm KernelDescManager Class
 * @param[in] imfea extracted feature
 * @param[out] dec_values socres from Support Vector machine
 */
void PBM::getScore( KernelDescManager& kdm, MatrixXf* imfea, double** dec_values )
{
#if MULTITHREAD_PART  
  omp_set_num_threads(m_pdNum);
#pragma omp parallel for
#endif
  for( int looppd = 0; looppd < m_pdNum; looppd++ ){
    
    getScore_liblinear( kdm, imfea[looppd], dec_values[looppd], looppd);
    //getScore_libsvm(); //sorry, not yet TODO!!
    
  }
}

/**
 * 各Part-Detectorからスコアを取得(線形SVM)
 * @return void
 * @param[in] kdm KernelDescManager Class
 * @param[in] imfea extracted feature
 * @param[out] dec_values socres from Support Vector machine
 * @param[in] pdIdx indicate the part-Detector No.
 */
void PBM::getScore_liblinear(KernelDescManager& kdm, MatrixXf& imfea, double* dec_values, int pdIdx )
{
  //Normalizationn
  MatrixXf minvalue = m_pDetector[pdIdx].minvalue;
  MatrixXf maxvalue = m_pDetector[pdIdx].maxvalue;
  MatrixXf imfea_s = kdm.scaletest_linear( imfea,minvalue, maxvalue);
  
  //liblinear形式へ変換
  int nr_class = get_nr_class( m_pDetector[pdIdx].svmModel );
  int nr_feature = get_nr_feature( m_pDetector[pdIdx].svmModel );
  struct feature_node *x = (struct feature_node *) malloc( (imfea_s.rows()+1) * sizeof(struct feature_node) );
  for( int i = 0; i < imfea_s.rows(); i++ ){
    x[i].index = i+1;
    x[i].value = imfea_s( i, 0 );
  }
  x[imfea_s.rows()].index = -1;
  
  if( m_pDetector[pdIdx].svmModel->bias >= 0 ){
    std::cerr << "model->bias >= 0! Not supported!!" << std::endl;
  }
  
  double predict_label = predict_values( m_pDetector[pdIdx].svmModel, x, dec_values );
  std::cerr << "Part-Detector No." << pdIdx+1 << " Predict Lable " << predict_label << std::endl;//debug
  free(x);
}


/***************************************************************
 * 2013/02/01 Written by Hideshi T. @DHRC
 *
 * RGB-D Object Recognition System based on Part-based Model
 * 
 * This is ported code from Matlab Platform
 **************************************************************/

#include <iostream>

#include "pbm.h"
#include "kerneldesc.h"
#include "detector.h"
#include "matio.h"
#include "func.h"

/****************************************************************
 *指定フォルダ中の画像(rgb,dep,loc)を読み取り順次認識を行う
 ***************************************************************/
void DataSetDemo( char** argv )
{
  std::string rootImgPath( argv[2] );
  
  cvNamedWindow("Input Image", CV_WINDOW_AUTOSIZE);
  CvFont font; SetupFont(font);
  
  PBM pbm(argv[1]);
  KernelDescManager kdm;
  
  std::vector<std::string> fileList;
  GetFileListFromDirectory( fileList, rootImgPath.c_str() );
  
  std::vector<std::string>::iterator itr;
  for( itr = fileList.begin(); itr < fileList.end(); ++itr ){
    std::cerr << "Image " << (*itr).c_str() << std::endl;
    
    IplImage* rgb = cvLoadImage( (rootImgPath + *itr).c_str(), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH );
    ++itr;
    IplImage* dep = cvLoadImage( (rootImgPath + *itr).c_str(), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH );
    ++itr;
    MatrixXf top_left = read_topleft_loc( (rootImgPath + *itr).c_str() );
    
    //Main Function
    double resultLable = pbm.Process(kdm, rgb, dep, top_left);
    std::string resultObj = pbm.getObjName(resultLable-1);
        
    //Show Result
    cvPutText(rgb, resultObj.c_str(), cvPoint(10,20), &font,cvScalar(0,256,0));
    cvShowImage("Input Image", rgb);
    cvWaitKey(0);
    
    //Release
    cvReleaseImage(&rgb);
    cvReleaseImage(&dep);
  }
  cvDestroyAllWindows();
}


void DataSetDemo_Segmentation( char** argv )
{
  std::string rootImgPath( argv[2] );
  
  cvNamedWindow("Input Image", CV_WINDOW_AUTOSIZE);
  CvFont font; SetupFont(font);
  
  PBM pbm(argv[1]);
  KernelDescManager kdm;
  Detector detector;
  
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc_pcl( new pcl::PointCloud<pcl::PointXYZRGBA> );
  std::vector<Eigen::MatrixXf> top_left; std::vector<CvPoint> bbox2d; std::vector<Eigen::Vector4f> bbox3d; 
  
  std::vector<std::string> fileList;
  GetFileListFromDirectory( fileList, rootImgPath.c_str() );
  
  std::vector<std::string>::iterator itr;
  for( itr = fileList.begin(); itr < fileList.end(); ++itr ){
    std::cerr << "Image " << (*itr).c_str() << std::endl;
    
    IplImage* rgb = cvLoadImage( (rootImgPath + *itr).c_str(), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH );
    ++itr;
    std::cerr << "Image " << (*itr).c_str() << std::endl;
    IplImage* dep = cvLoadImage( (rootImgPath + *itr).c_str(), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH );
    ++itr;
    
    IplImage* pc = cvCreateImage( cvSize(dep->width, dep->height), IPL_DEPTH_32F, 1 );
    
    //Main Function
    depth2cloud( &kdm, dep, pc );
    detector.setpcl( pc, pc_pcl );
    detector.detect( dep, pc_pcl, top_left, bbox2d, bbox3d );
    
    for( int i = 0; i < bbox2d.size(); i+=2 ){
      cv::Rect brect( (int)bbox2d[i].x, (int)bbox2d[i].y, (int)bbox2d[i+1].x-(int)bbox2d[i+0].x, (int)bbox2d[i+1].y-(int)bbox2d[i+0].y );
      
      cvSetImageROI( rgb, brect );
      cvSetImageROI( dep, brect );
      IplImage* rgb_crop = cvCreateImage( cvSize(brect.width, brect.height), IPL_DEPTH_8U, 3 );
      IplImage* dep_crop = cvCreateImage( cvSize(brect.width, brect.height), IPL_DEPTH_16U, 1 );
      cvCopy( rgb, rgb_crop, NULL );
      cvResetImageROI( rgb );
      cvCopy( dep, dep_crop, NULL );
      cvResetImageROI( dep );
      
      double resultLable = pbm.Process(kdm, rgb_crop, dep_crop, top_left[i/2]);
      std::string resultObj = pbm.getObjName(resultLable-1);
      
      cvRectangle( rgb, cvPoint( (int)bbox2d[i].x, (int)bbox2d[i].y ),
		   cvPoint( (int)bbox2d[i+1].x, (int)bbox2d[i+1].y ),
		   cvScalar( 0, 0, 255 ), 2, 8, 0 );
      cvPutText(rgb, resultObj.c_str(), cvPoint(bbox2d[i].x,bbox2d[i].y), &font,cvScalar(0,256,0));
      cvReleaseImage(&rgb_crop);
      cvReleaseImage(&dep_crop);
    }
    
    //Show Result
    cvShowImage("Input Image", rgb);
    cvWaitKey(0);
    
    //Release
    cvReleaseImage(&rgb);
    cvReleaseImage(&dep);
  }
  cvDestroyAllWindows();
}

/****************************************************************
 *
 ***************************************************************/
IplImage* frame = NULL; IplImage* frame_dep = NULL; IplImage* frame_pc = NULL;
boost::mutex m; bool isRun = true;
void ThreadSegmentation()
{
  Detector detector;
  
  IplImage* rgb_src = NULL; IplImage* dep_src = NULL; IplImage* pc_src = NULL;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc_pcl( new pcl::PointCloud<pcl::PointXYZRGBA> );
  std::vector<Eigen::MatrixXf> top_left;
  std::vector<CvPoint> bbox2d; std::vector<Eigen::Vector4f> bbox3d; 
  
  bool isGetImg = false;
  cvNamedWindow("Process View", CV_WINDOW_AUTOSIZE);
  cvMoveWindow("Process View", 50, 50);
  
  while( isRun )
    {
      top_left.clear(); bbox2d.clear(); bbox3d.clear();
      {
	boost::mutex::scoped_lock lock(m);
	if( frame != NULL && frame_dep != NULL && frame_pc != NULL ){
	  cvReleaseImage(&rgb_src);
	  cvReleaseImage(&dep_src);
	  cvReleaseImage(&pc_src);
	  rgb_src = cvCloneImage(frame);
	  dep_src = cvCloneImage(frame_dep);
	  pc_src = cvCloneImage(frame_pc);
	  
	  isGetImg = true;
	}
	
      }
      
      if( isGetImg ){
	detector.setpcl( pc_src, pc_pcl );
	detector.detect( dep_src, pc_pcl, top_left, bbox2d, bbox3d );
	isGetImg = false;
      }
      
      for( int i = 0; i < bbox2d.size(); i+=2 ){
	std::cout << bbox2d[i].x << " " << bbox2d[i].y << " : " << bbox2d[i+1].x << " " << bbox2d[i+1].y << std::endl;
	cvRectangle( rgb_src, cvPoint( (int)bbox2d[i].x, (int)bbox2d[i].y ),
		     cvPoint( (int)bbox2d[i+1].x, (int)bbox2d[i+1].y ),
		     cvScalar( 0, 0, 255 ), 2, 8, 0 );
      }
      cvShowImage("Process View", rgb_src);
    }
}

void ThreadLive( char* model )
{
  CvFont font; SetupFont(font);
  CvFont fpsFont; cvInitFont(&fpsFont,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 0.7, 0.7, 0, 1);
  
  PBM pbm(model);
  KernelDescManager kdm;
  Detector detector;
  
  IplImage* rgb_src = NULL; IplImage* dep_src = NULL; IplImage* pc_src = NULL;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc_pcl( new pcl::PointCloud<pcl::PointXYZRGBA> );
  std::vector<Eigen::MatrixXf> top_left;
  std::vector<CvPoint> bbox2d; std::vector<Eigen::Vector4f> bbox3d; 
  
  bool isGetImg = false;
  MouseParam mparam;
  cvNamedWindow("Process View", CV_WINDOW_AUTOSIZE);
  cvMoveWindow("Process View", 50, 50);
  cv::setMouseCallback("Process View", &mouseFunc, &mparam);
  Timer t;
  
  while( isRun )
    {
      
      top_left.clear(); bbox2d.clear(); bbox3d.clear();
      {
	boost::mutex::scoped_lock lock(m);
	if( frame != NULL && frame_dep != NULL && frame_pc != NULL ){
	  cvReleaseImage(&rgb_src);
	  cvReleaseImage(&dep_src);
	  cvReleaseImage(&pc_src);
	  rgb_src = cvCloneImage(frame);
	  dep_src = cvCloneImage(frame_dep);
	  pc_src = cvCloneImage(frame_pc);
	  
	  isGetImg = true;
	}
	
      }
      
      if( isGetImg ){
	t.start();
	
	detector.setpcl( pc_src, pc_pcl );
	detector.detect( dep_src, pc_pcl, top_left, bbox2d, bbox3d );
	
	//Recognition Loop
	for( int i = 0; i < bbox2d.size(); i+=2 ){
	  cv::Rect brect( (int)bbox2d[i].x, (int)bbox2d[i].y, (int)bbox2d[i+1].x-(int)bbox2d[i+0].x, (int)bbox2d[i+1].y-(int)bbox2d[i+0].y );
	  
	  cvSetImageROI( rgb_src, brect );
	  cvSetImageROI( dep_src, brect );
	  IplImage* rgb_crop = cvCreateImage( cvSize(brect.width, brect.height), IPL_DEPTH_8U, 3 );
	  IplImage* dep_crop = cvCreateImage( cvSize(brect.width, brect.height), IPL_DEPTH_16U, 1 );
	  cvCopy( rgb_src, rgb_crop, NULL );
	  cvResetImageROI( rgb_src );
	  cvCopy( dep_src, dep_crop, NULL );
	  cvResetImageROI( dep_src );
	  
	  double resultLable = pbm.Process(kdm, rgb_crop, dep_crop, top_left[i/2]);
	  std::string resultObj = pbm.getObjName(resultLable-1);
	  
	  cvRectangle( rgb_src, cvPoint( (int)bbox2d[i].x, (int)bbox2d[i].y ),
		       cvPoint( (int)bbox2d[i+1].x, (int)bbox2d[i+1].y ),
		       cvScalar( 0, 0, 255 ), 2, 8, 0 );
	  cvPutText(rgb_src, resultObj.c_str(), cvPoint(bbox2d[i].x,bbox2d[i].y), &font,cvScalar(0,256,0));
	  cvReleaseImage(&rgb_crop);
	  cvReleaseImage(&dep_crop);
	}
	
	char fps[16];
	sprintf( fps, "%.3lf [sec]", t.get() );
	cvPutText(rgb_src, fps, cvPoint(rgb_src->width-150,30), &font, cvScalar(0,0,256));
	cvShowImage("Process View", rgb_src);

	isGetImg = false;
      }
      
    }
}


void SegmentationDemo()
{
  boost::thread workerThread(&ThreadCaptureFrame);
  boost::thread detectorThread(&ThreadSegmentation);
  workerThread.join();
  detectorThread.join();
}

void LiveDemo(char* model)
{
  boost::thread workerThread(&ThreadCaptureFrame);
  boost::thread RecognitionThread(boost::bind(&ThreadLive, model));
  workerThread.join();
  RecognitionThread.join();
}

int main( int argc, char** argv )
{
  int opt = -1;
  std::cout << "Enter  '1' for Simple Image Demo, not yet TODO!!,          " << std::endl
	    << "       '2' for Dataset Demo,                               " << std::endl
	    << "       '3' for Dataset Demo(Segmentation), not yet TODO!!, " << std::endl
	    << "       '4' for Segmentation Demo (No Recognition),         " << std::endl
	    << "       '5' for Live Recognition Demo(Segmentation).        " << std::endl;
  std::cout << "--->";
  std::cin >> opt;
  std::cout << "Opt: " << opt << std::endl;
  
  switch( opt ){

  case 1:
    //Simple Demo
    //ImageDemo( argv ); //Not yet!!
    break;
  
  case 2:
    //Dataset Demo
    DataSetDemo( argv );
    break;

  case 3:
    //Dataset Demo
    DataSetDemo_Segmentation( argv );//Not yet!!
    break;

  case 4:
    //Segmentation Demo (No Recognition)
    SegmentationDemo();
    break;

  case 5:
    //Live(Kinect) Demo
    LiveDemo(argv[1]);
    break;

  default:
    std::cerr << "No Option!!" << std::endl;
    break;
    
  }
  
  return 0;
}

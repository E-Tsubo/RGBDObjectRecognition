#ifndef __FUNC__
#define __FUNC__

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "kerneldesc.h"

cv::VideoCapture captureKinect( CV_CAP_OPENNI );
extern IplImage* frame;
extern IplImage* frame_dep;
extern IplImage* frame_pc;
static IplImage* out = NULL;
static IplImage* out_dep = NULL;
static IplImage* out_pc = NULL;
extern boost::mutex m;
extern bool isRun;

//マウスの状態を格納する構造体
typedef struct MouseParam{
  unsigned int x;
  unsigned int y;
  int event;
  int flags;
} MouseParam;

//Callback関数[マウスの状態を取得]
void mouseFunc(int event, int x, int y, int flags, void  *param)
{
  MouseParam *mparam = (MouseParam*)param;
  mparam->x = x;
  mparam->y = y;
  mparam->event = event;
  mparam->flags = flags;
  //std::cerr << "Mouse Pos:" << x << " " << y << std::endl;//debug
}

void GetFileListFromDirectory(vector<string>& src, const char* directory)
{
  if (boost::filesystem::is_directory(directory))
    {
      for (boost::filesystem::directory_iterator itr(directory);
	   itr != boost::filesystem::directory_iterator(); ++itr)
	{
	  if (!is_directory(itr->status()))
	    {
	      //string fn = itr->path().filename().string(); // new boost version
	      string fn = itr->path().filename(); // old boost version
	      src.push_back(fn);
	      //std::cout << fn << std::endl;//debug
	    }
	  
	}
      
    } else {
    cout << "Image directory not found in path!" << endl;
  }
  
  //Sorting
  std::sort(src.begin(), src.end());
  //for( int i = 0; i < src.size(); i++ ) std::cout << src[i] << std::endl;//debug
}

MatrixXf read_topleft_loc( string filename )
{
  std::ifstream instream;
  string ss;
  MatrixXf top_left(2,1);
  
  if (!boost::filesystem::exists(filename)) {
    cout << "warning: " << filename << " not found; using top_left=(1,1)" << endl;
    top_left.fill(1.0f);
    return top_left;
  }
  
  instream.open( filename.c_str() );
  getline( instream, ss );
  int pos=ss.find(",");
  top_left(0)=atoi( ss.substr(0,pos).c_str() );
  top_left(1)=atoi( ss.substr(pos+1).c_str() );
  
  instream.close();
  
  return top_left;
}

void SetupFont (CvFont& font)
{
  double hScale=0.7;
  double vScale=0.7;
  int    lineWidth=1;
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
}

void kinectCapture()
{
  IplImage tmp, tmp_dep, tmp_pc;
  cv::Mat kinimg, kinimg_dep, kinimg_pc;
    
  captureKinect.grab();
  captureKinect.retrieve( kinimg, CV_CAP_OPENNI_BGR_IMAGE );//CV_8UC3
  captureKinect.retrieve( kinimg_dep, CV_CAP_OPENNI_DEPTH_MAP );//CV_16UC1
  captureKinect.retrieve( kinimg_pc, CV_CAP_OPENNI_POINT_CLOUD_MAP );//CV_32FC3
  
  for( int y = 0; y < kinimg_pc.rows; y++ ){
    for( int x = 0; x < kinimg_pc.cols; x++ ){
      cv::Point3f &p = ((cv::Point3f*)(kinimg_pc.data+kinimg_pc.step.p[0]*y))[x];
      //std::cout << p.x << " " << p.y << " " << p.z << std::endl;
    }
  }
  
  //Sharing the image data. not copy.
  tmp = kinimg; tmp_dep = kinimg_dep; tmp_pc = kinimg_pc;
  
  cvReleaseImage( &out );
  cvReleaseImage( &out_dep );
  cvReleaseImage( &out_pc );
  out = cvCreateImage( cvSize( tmp.width, tmp.height ), IPL_DEPTH_8U, 3 );
  out_dep = cvCreateImage( cvSize( tmp.width, tmp.height ), IPL_DEPTH_16U, 1 );
  out_pc = cvCreateImage( cvSize( tmp.width, tmp.height ), IPL_DEPTH_32F, 3 );
  cvCopy( &tmp, out, NULL );
  cvCopy( &tmp_dep, out_dep, NULL );
  cvCopy( &tmp_pc, out_pc, NULL );
  
  //Sharing the image data. not copy.
  frame = out; frame_dep = out_dep; frame_pc = out_pc;
}

void ThreadCaptureFrame()
{
  std::cout << "Cam capture started..." << std::endl;
  
  if(!captureKinect.isOpened()){
    std::cerr << "can't open Kinect" << std::endl;
    exit(-1);
  }
  //CvFont fpsFont;
  //cvInitFont(&fpsFont, CV_FONT_HERSHEY_SIMPLEX , 1.0f, 1.0f, 0, 1, CV_AA);
  //char fps[32] = "15";
  
  int key = 0;
  
  //cvNamedWindow("Camera View", CV_WINDOW_AUTOSIZE);
  //cvMoveWindow("Camera View", 50, 50);
  
  while ((char)key != 'q' & 0xff)
    {
      {
	boost::mutex::scoped_lock lock(m);
	kinectCapture();
      }
      
      if (!frame) break;
      
      //cvPutText(frame, fps, cvPoint(10,450), &fpsFont, cvScalar(0,0,255));
      //cvShowImage("Camera View", frame);
      key = cvWaitKey(10);
    }
  
  cvReleaseImage(&frame);
  cvReleaseImage(&frame_dep);
  
  captureKinect.release();
  std::cout << "Cam capture ended" << std::endl;
  isRun = false;
}

void depth2cloud(KernelDescManager* kdm, IplImage* dep, IplImage* pc)
{
  int height=dep->height;
  int width=dep->width;
  
  Eigen::MatrixXf depth(height,width);
  Eigen::MatrixXf tmp(2,1); tmp(0,0) = 1.0; tmp(1,0) = 1.0;
  
  CvScalar s;
  for( int y = 0; y < height; y++ ){
    for( int x = 0; x < width; x++ ){
      s = cvGet2D( dep, y, x );
      depth(y,x) = s.val[0];
    }
  }
  
  Eigen::MatrixXf pcloud_x, pcloud_y, pcloud_z;
  kdm->depth2cloud( depth, pcloud_x, pcloud_y, pcloud_z, tmp );
  
  for( int y = 0; y < height; y++ ){
    for( int x = 0; x < width; x++ ){
      reinterpret_cast<float *>(pc->imageData + y*pc->widthStep/sizeof(float))[x*pc->nChannels+0] = pcloud_x(y,x)/1000.0;//mm->m
      reinterpret_cast<float *>(pc->imageData + y*pc->widthStep/sizeof(float))[x*pc->nChannels+1] = pcloud_y(y,x)/1000.0;
      reinterpret_cast<float *>(pc->imageData + y*pc->widthStep/sizeof(float))[x*pc->nChannels+2] = pcloud_z(y,x)/1000.0;
    }
  }
}
#endif

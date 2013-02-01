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

int main( int argc, char** argv )
{
  //Simple Demo
  //ImageDemo( argv );
  
  //Dataset Demo
  DataSetDemo( argv );
  
  //Live(Kinect) Demo
  
  return 0;
}

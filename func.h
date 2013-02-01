#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

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
  double hScale=0.4;
  double vScale=0.4;
  int    lineWidth=1;
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
}

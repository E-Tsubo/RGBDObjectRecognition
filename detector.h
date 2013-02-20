#ifndef __DETECTOR__
#define __DETECTOR__

#include <vector>
#include <numeric>
#include <algorithm>
#include <boost/timer.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>  
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
//#include <pcl/io/openni_grabber.h>  
//#include <pcl/kdtree/kdtree.h>
//#include <pcl/sample_consensus/method_types.h>  
//#include <pcl/sample_consensus/model_types.h>  
//#include <pcl/features/normal_3d.h>

class Detector
{
 public:
  Detector();
  ~Detector();
  
  void setpcl( IplImage* pc, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcl_pc );
  void detect( IplImage* dep,
	       pcl::PointCloud<pcl::PointXYZRGBA>::Ptr,
	       std::vector<Eigen::MatrixXf>& topleft,
	       std::vector<CvPoint>& bbox2d,
	       std::vector<Eigen::Vector4f>& bbox3d );
  
  void planeSeg( pcl::PointCloud<pcl::PointXYZRGBA>& cloud,
		 pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filtered,
		 double threshould );
  
  void cluster( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr coloredCloud,
		std::vector< pcl::PointCloud<pcl::PointXYZRGBA> >& divide );
  
  void bbox( std::vector< pcl::PointCloud<pcl::PointXYZRGBA> >& divide,
	     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
	     std::vector<Eigen::Vector4f>& bbox3d );
  
  void bbox2dbbox( IplImage* dep,
		   std::vector<Eigen::Vector4f>& bbox3d,
		   std::vector<CvPoint>& bbox2d );
  
  void evalbboxPos( std::vector<CvPoint>& bbox2d, IplImage* dep );
  
  void sortDepth( std::vector<Eigen::Vector4f>& bbox3d, 
		  std::vector<CvPoint>& bbox2d );
  void setTopleftPos( std::vector<CvPoint>& bbox2d,
		      std::vector<Eigen::MatrixXf>& topleft );
 private:
  
};


class compare
{
private:
  std::vector<double> const& m_v;
  
 public:
  typedef bool result_type;
    
 compare(std::vector<double> const& _v)
   : m_v(_v)
  {}
  
  bool operator()(std::size_t a, std::size_t b) const
  {
    return (m_v[a] < m_v[b]);
  }
};
#endif

#include "detector.h"

Detector::Detector()
{

}

Detector::~Detector()
{

}

//OpenNI Grabberより得られた3次元座標を格納した画像データ->pc
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Detector::setpcl( IplImage* pc )
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp( new pcl::PointCloud<pcl::PointXYZRGBA> );
  tmp->width = pc->width; tmp->height = pc->height;
  tmp->resize( tmp->width * tmp->height );
  
  //fast access implimentation
  for( int y = 0; y < pc->height; y++ ){
    for( int x = 0; x < pc->width; x++ ){
      int a = pc->widthStep*y+(x*3);
      tmp->points[y*pc->width+x].x = pc->imageData[a+0];
      tmp->points[y*pc->width+x].y = pc->imageData[a+1];
      tmp->points[y*pc->width+x].z = pc->imageData[a+2];
    }
  }
  
  return tmp;
}

void Detector::detect(IplImage* dep, 
		      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
		      std::vector<Eigen::MatrixXf>& topleft,
		      std::vector<CvPoint>& bbox2d,
		      std::vector<Eigen::Vector4f>& bbox3d )
{
  pcl::PointCloud<pcl::PointXYZRGBA> target(*cloud);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr segmented( new pcl::PointCloud<pcl::PointXYZRGBA> );
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colored( new pcl::PointCloud<pcl::PointXYZRGBA> );//for debug
  std::vector< pcl::PointCloud<pcl::PointXYZRGBA> > divide;
  
  planeSeg( target, segmented, 0.01 );
  cluster( segmented, colored, divide );
  bbox( divide, colored, bbox3d );
  bbox2dbbox( dep, bbox3d, bbox2d );
  setTopleftPos( bbox2d, topleft );
}

void Detector::planeSeg(pcl::PointCloud<pcl::PointXYZRGBA>& cloud,
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered,
			//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr clustered,
			//std::vector< pcl::PointCloud<pcl::PointXYZRGBA> >& divide,
			double threshould )
{
  boost::timer t;
  
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);  
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);  
  
  //delete nan data nad not near data from kinect raw data
  pcl::PassThrough<pcl::PointXYZRGBA> pass;
  pass.setInputCloud( cloud.makeShared() );//makeShared provide smartPtr.
  pass.setFilterFieldName( "z" );
  pass.setFilterLimits( 0.0, 0.8 );
  pass.filter( cloud );

  //Down Sampling
  pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
  sor.setInputCloud( cloud.makeShared() );
  sor.setLeafSize( 0.005f, 0.005f, 0.002f );
  sor.filter( cloud );
  
  std::cout << "Down Sampling:" << t.elapsed() << " sec" << std::endl;
  t.restart();
  
  
  // Create the segmentation object  
  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;  
  // Optional  
  seg.setOptimizeCoefficients (true);  
  // Mandatory  
  seg.setModelType (pcl::SACMODEL_PLANE);  
  seg.setMethodType (pcl::SAC_RANSAC);  
  seg.setDistanceThreshold (threshould);  
  
  seg.setInputCloud (cloud.makeShared ());  
  seg.segment (*inliers, *coefficients);
  
  // Colored the plane part for debug
  for (size_t i = 0; i < inliers->indices.size (); ++i) {  
    cloud.points[inliers->indices[i]].r = 255;  
    cloud.points[inliers->indices[i]].g = 0;  
    cloud.points[inliers->indices[i]].b = 0;  
  }
  
  
  // Extract the rest part
  pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
  
  //pcl::PointCloud<pcl::PointXYZRGBA> filter;
  //pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
  //cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBA>);
  
  extract.setInputCloud( cloud.makeShared() );
  extract.setIndices( inliers );
  //true:delete plane part, false:delete not plane part.
  extract.setNegative( true );
  //extract.filter( cloud );
  extract.filter( *cloud_filtered );
  
  std::cout << "Plane Segmentation:" << t.elapsed() << " sec" << std::endl;
  t.restart();
  
}

void Detector::cluster( pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colored,
			std::vector< pcl::PointCloud<pcl::PointXYZRGBA> >& divide )
{
  boost::timer t;
  
  //Clustering
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr
    tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
  tree->setInputCloud( cloud );
  
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
  ec.setClusterTolerance( 0.02 );
  ec.setMinClusterSize( 500 );
  ec.setMaxClusterSize( 10000 );
  ec.setSearchMethod( tree );
  ec.setInputCloud( cloud );
  ec.extract( cluster_indices );
  
  int j = 0;
  for( std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
       it != cluster_indices.end();
       ++it ){
    pcl::PointCloud<pcl::PointXYZRGBA> cloud_cluster;
    
    for( std::vector<int>::const_iterator pit = it->indices.begin();
	 pit != it->indices.end();
	 pit++ ){
      
      cloud_cluster.points.push_back( cloud->points[*pit] );
      
      //coloring in order to show segmentation object for debug
      cloud->points[*pit].r = colors[j%10][0];
      cloud->points[*pit].g = colors[j%10][1];
      cloud->points[*pit].b = colors[j%10][2];
      colored->points.push_back( cloud->points[*pit] );
    }
    
    //Store Native Data per detected objects
    cloud_cluster.width = cloud_cluster.points.size();
    cloud_cluster.height = 1;
    cloud_cluster.is_dense = true;
    
    divide.push_back( cloud_cluster );
    
    //Store Coloring Data for all segmentation object
    colored->width = colored->points.size();
    colored->height = 1;
    colored->is_dense = true;
    
    j++;
  }
  std::cout << "Clustering:" << t.elapsed() << " sec" << std::endl << std::endl;
}

void Detector::bbox( std::vector< pcl::PointCloud<pcl::PointXYZRGBA> >& divide,
		     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
		     std::vector<Eigen::Vector4f>& bbox3d )
{
  Eigen::Vector4f min_point;
  Eigen::Vector4f max_point;

  for( int i = 0; i < divide.size(); i++ ){
    
    pcl::getMinMax3D( divide[i], min_point, max_point );
    
    pcl::PointXYZRGBA tmp1, tmp2;
    tmp1.x = min_point.x(); tmp1.y = min_point.y(); tmp1.z = min_point.z();
    tmp1.r = 255; tmp1.g = 255; tmp1.b = 0;
    tmp2.x = max_point.x(); tmp2.y = max_point.y(); tmp2.z = max_point.z();
    tmp2.r = 255; tmp2.g = 255; tmp2.b = 0;
    
    cloud->push_back(tmp1);
    cloud->push_back(tmp2);//バウンディングボックスを示す端点をPointCloudへ追加
    
    bbox3d.push_back( min_point );
    bbox3d.push_back( max_point );
    
  }
}

void Detector::bbox2dbbox( IplImage* dep,
			   std::vector<Eigen::Vector4f>& bbox3d,
			   std::vector<CvPoint>& bbox2d )
{
  //Convert 3D Point Cloud to 2D Image.
  const unsigned short* depthImageMapPtr = (unsigned short*)dep->imageData;
  
  //Kinect Sensor Parameter
  int width = dep->width;
  int height = dep->height;
  float center_x = width/2; float center_y = height/2;
  float focal_length_x = 600.0;
  float focal_length_y = 600.0;
  float desiredAngularResolution = asinf( 0.5f * float(640)/float(focal_length_x) ) / (0.5f*float(640));
  
  pcl::RangeImagePlanar range_image_planar;
  range_image_planar.setDepthImage( depthImageMapPtr, width, height, center_x, center_y,
				    focal_length_x, focal_length_y, desiredAngularResolution );
  
  for( int i = 0; i < bbox3d.size(); i += 2 ){
    Eigen::Vector3f tmp1( bbox3d[i+0].x(), bbox3d[i+0].y(), bbox3d[i+0].z() );
    Eigen::Vector3f tmp2( bbox3d[i+1].x(), bbox3d[i+1].y(), bbox3d[i+1].z() );
    
    int x, y;
    float range;
    CvPoint tmp_pt2d;
    
    range_image_planar.getImagePoint( tmp1, x, y, range);
    tmp_pt2d.x = x; tmp_pt2d.y = y;
    bbox2d.push_back( tmp_pt2d );
		
    range_image_planar.getImagePoint( tmp2, x, y, range);
    tmp_pt2d.x = x; tmp_pt2d.y = y;
    bbox2d.push_back( tmp_pt2d );
    
    //座標系変換
    bbox2d[i].y = height - bbox2d[i].y;
    bbox2d[i+1].y = height - bbox2d[i+1].y;
    if( bbox2d[i].x > bbox2d[i+1].x ){
      int tmp = bbox2d[i].x;
      bbox2d[i].x = bbox2d[i+1].x;
      bbox2d[i+1].x = tmp;
    }
    if( bbox2d[i].y > bbox2d[i+1].y ){
      int tmp = bbox2d[i].y;
      bbox2d[i].y = bbox2d[i+1].y;
      bbox2d[i+1].y = tmp;
    }
  }
  
}


void Detector::setTopleftPos( std::vector<CvPoint>& bbox2d,
			      std::vector<Eigen::MatrixXf>& topleft )
{
  Eigen::MatrixXf tmp;
  for( int i = 0; i < bbox2d.size(); i+=2 ){
    //bbox2はバウンディングボックスの左上点、右下点が格納されている
    //bbox2dbbox関数にて[i],[i+1]の内[i]に左上点が格納されているのが保証されている
    //そのため、本関数内でチェックはない
    //Spin Image Featureにて利用されるdepth2cloud関数ではtop_left座標を一律 -1 している
    //これはMatlab上では画像が(1,1)から始まるからである OpenCV等では(0,0)から始まるので注意
    //ここではそのオフセットとしてmatlab座標系に変換している
    //なお、(0)にy座標の値、(1)にx座標が格納されているので注意 これもmatlab座標系の名残
    tmp(0) = bbox2d[i].y+1;
    tmp(1) = bbox2d[i].x+1;
    
    topleft.push_back(tmp);
  }
}

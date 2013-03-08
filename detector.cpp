#include "detector.h"

float colors[10][3] = { { 0, 255, 0 },
			{ 0, 0, 255 },
			{ 255, 0, 255 },
			{ 0, 255, 255 },
			{ 255, 255, 0 },
			{ 125, 125, 125 },
			{ 125, 0, 255 },
			{ 255, 125, 0 },
			{ 255, 0, 125 },
			{ 0, 125, 255 } };

Detector::Detector()
{

}

Detector::~Detector()
{

}

/**
 * OpenNI Grabberより得られた3次元座標を格納した画像データ->pc
 * @param[in] pc point cloud data from kinect
 * @param[out] tmp convert to pcl format
 */
void Detector::setpcl( IplImage* pc, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp)
{
  tmp->width = pc->width; tmp->height = pc->height;
  tmp->resize( tmp->width * tmp->height );
  
  //キャスト回数を減らして高速化
  int step = pc->widthStep/sizeof(float);
  float *pData = reinterpret_cast<float *>(pc->imageData);
  
  for( int y = 0; y < pc->height; y++ ){
    for( int x = 0; x < pc->width; x++ ){
      //tmp->points[y*pc->width+x].x = reinterpret_cast<float *>(pc->imageData + y*pc->widthStep)[x*pc->nChannels];
      //tmp->points[y*pc->width+x].y = reinterpret_cast<float *>(pc->imageData + y*pc->widthStep)[x*pc->nChannels+1];
      //tmp->points[y*pc->width+x].y = reinterpret_cast<float *>(pc->imageData + y*pc->widthStep)[x*pc->nChannels+2];
      
      //Fast implementation
      tmp->points[y*pc->width+x].x = pData[y*step+x*pc->nChannels+0];
      tmp->points[y*pc->width+x].y = pData[y*step+x*pc->nChannels+1];
      tmp->points[y*pc->width+x].z = pData[y*step+x*pc->nChannels+2];
    }
  }
  
}

/**
 * メイン関数 点群データから卓上物体を検出
 * @param[in] dep Depth Image from kinect
 * @param[in] cloud point cloud from kinect
 * @param[in] topleft cropped image's pos
 * @param[out] bbox2d the segmented result(2D)
 * @param[out] bbox3d the segmented result(3D)
 */
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
  if( divide.size() == 0 )
    return;
  bbox( divide, colored, bbox3d );
  bbox2dbbox( dep, bbox3d, bbox2d );
  evalbboxPos( bbox2d, dep );
  sortDepth( bbox3d, bbox2d );
  setTopleftPos( bbox2d, topleft );
}

/**
 * 平面検出及び削除
 * @param[in] cloud input point cloud data
 * @param[out] cloud_filtered point cloud without plane part
 * @param[in] threshould
 */
void Detector::planeSeg(pcl::PointCloud<pcl::PointXYZRGBA>& cloud,
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered,
			double threshould )
{
  boost::timer t;
  
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);  
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);  
  
  //delete nan data and not near data from kinect raw data
  pcl::PassThrough<pcl::PointXYZRGBA> pass;
  pass.setInputCloud( cloud.makeShared() );//makeShared provide smartPtr.
  pass.setFilterFieldName( "z" );
  pass.setFilterLimits( 0.0, 0.7 );//奥行き制限
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
  extract.filter( *cloud_filtered );
  
  std::cout << "Plane Segmentation:" << t.elapsed() << " sec" << std::endl;
  t.restart();
}

/**
 * 物体を点群の距離をもとに階層型クラスタリング
 * @param[in] cloud input point cloud data
 * @param[out] colored for visualization
 * @param[out] divide clustered objects
 */
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

/**
 * バウンディングボックス作成
 * @param[in] clustered object
 * @param[out] cloud add bounding box points
 * @param[out] bbox3d bounding box's position
 */
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

/**
 * ３次元バウンディングボックスから２次元バウンディングボックスを作成
 * @param[in] dep depth image from kinect
 * @param[in] bbox3d 
 * @param[out] bbox2d 
 */
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
  float focal_length_x = 570.3;//525.0;//600.0;
  float focal_length_y = 570.3;//525.0;//600.0;
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

/**
 * 画面上からバウンディングボックスがはみ出していないか確認及び修正
 */
void Detector::evalbboxPos( std::vector<CvPoint>& bbox2d, IplImage* dep )
{
  int width = dep->width - 1; int height = dep->height - 1;
  for( std::vector<CvPoint>::iterator itr = bbox2d.begin();
       itr != bbox2d.end(); itr++ ){
    if( itr->x < 0 )
      itr->x = 0;
    else if( itr->x > width )
      itr->x = width;
    
    if( itr->y < 0 )
      itr->y = 0;
    else if( itr->y > height )
      itr->y = height;
  }
}

/**
 * 距離情報を基にソート. 近い物体から順に認識を行うため
 */
void Detector::sortDepth( std::vector<Eigen::Vector4f>& bbox3d, 
			  std::vector<CvPoint>& bbox2d )
{
  std::vector<double> center_dep;
  
  for( std::vector<Eigen::Vector4f>::iterator itr = bbox3d.begin();
       itr != bbox3d.end(); itr++ ){
    double tmp = itr->z(); itr++;
    tmp += itr->z(); tmp /= 2.0;
    center_dep.push_back(tmp);
    //std::cout << tmp << std::endl;//debug
  }
  
  std::vector<std::size_t> idx(center_dep.size());
  //std::iota(idx.begin(), idx.end(), 0);//iotaが見つからない. Verによるのかも.再利用性を考慮してforで記述
  for( int i = 0; i < idx.size(); i++ )
    idx[i] = i*2;
  
  std::sort(idx.begin(), idx.end(), compare(center_dep) );
  
  std::vector<Eigen::Vector4f> tmp3; std::vector<CvPoint> tmp2;
  for( std::size_t i = 0; i < idx.size(); i++ ){
    tmp3.push_back( bbox3d[ idx[i] ] );
    tmp3.push_back( bbox3d[ idx[i]+1 ] );
    
    tmp2.push_back( bbox2d[ idx[i] ] );
    tmp2.push_back( bbox2d[ idx[i]+1 ] );
  }
  
  //for( std::size_t i = 0; i < idx.size()*2; i+=2 )
  //std::cout << (tmp3[i].z()+tmp3[i+1].z())/2.0 << std::endl;//debug
  bbox3d.swap( tmp3 );
  bbox2d.swap( tmp2 );
}

/**
 * セグメンテーションした画像の左上座標を保存
 * @param[in] bbox2d bounding box pos
 * @param[out] topleft cropped image pos
 */
void Detector::setTopleftPos( std::vector<CvPoint>& bbox2d,
			      std::vector<Eigen::MatrixXf>& topleft )
{
  //VectorXf is a dynamic-size vector of floats Matrix<float, Dynamic, 1>//
  Eigen::VectorXf tmp(2);
  for( int i = 0; i < bbox2d.size(); i+=2 ){
    //bbox2はバウンディングボックスの左上点、右下点が格納されている
    //bbox2dbbox関数にて[i],[i+1]の内[i]に左上点が格納されているのが保証されている
    //そのため、本関数内でチェックはない
    //Spin Image Featureにて利用されるdepth2cloud関数ではtop_left座標を一律 -1 している
    //これはMatlab上では画像が(1,1)から始まるからである OpenCV等では(0,0)から始まるので注意
    //ここではそのオフセットとしてmatlab座標系に変換している
    //なお、(0)にy座標の値、(1)にx座標が格納されているので注意 これもmatlab座標系の名残
    tmp[0] = bbox2d[i].y+1;
    tmp[1] = bbox2d[i].x+1;
    
    topleft.push_back(tmp);
  }
}

CXX	= g++
CFLAG	= -O3 -msse -msse2 -msse3 -msse4 -fopenmp -mtune=native -march=native
#CXX	= icc
#CFLAG	= -O3 -xHOST -ipo -msse -msse2 -msse3 -msse4 -fopenmp
PCL_FLAG= `pkg-config pcl_common-1.5 pcl_io-1.5 pcl_sample_consensus-1.5 pcl_filters-1.5 pcl_segmentation-1.5 pcl_range_image_border_extractor-1.5 --cflags --libs`

all: pbmrecog

pbmrecog: main.o pbm.o kerneldesc.o detector.o
	$(CXX) $(CFLAG) -o pbmrecog main.o pbm.o kerneldesc.o detector.o liblinear/tron.cpp liblinear/linear.cpp liblinear/blas/blas.a \
	`pkg-config opencv matio --cflags --libs` $(PCL_FLAG) -lhdf5 -IInclude -lboost_filesystem -lboost_system -lboost_thread

main.o: main.cpp
	$(CXX) $(CFLAG) -c main.cpp `pkg-config opencv matio --cflags --libs` $(PCL_FLAG) -lhdf5 -IInclude -lboost_filesystem -lboost_system -lboost_thread

pbm.o: pbm.cpp
	$(CXX) $(CFLAG) -c pbm.cpp `pkg-config opencv matio --cflags --libs` -lhdf5 -IInclude -lboost_filesystem -lboost_system

kerneldesc.o: kerneldesc.cpp
	$(CXX) $(CFLAG) -c kerneldesc.cpp `pkg-config opencv matio --cflags --libs` -lhdf5 -IInclude -lboost_filesystem -lboost_system

detector.o: detector.cpp
	$(CXX) $(CFLAG) -c detector.cpp -IInclude `pkg-config opencv --cflags --libs` $(PCL_FLAG)

clean:
	rm pbmrecog main.o pbm.o kerneldesc.o detector.o


#main: main.cpp
#	g++ -O3 -fopenmp main.cpp pbm.cpp kerneldesc.cpp liblinear/tron.cpp liblinear/linear.cpp liblinear/blas/blas.a -o main `pkg-config opencv matio --cflags --libs` -lhdf5 -IInclude -lboost_filesystem -lboost_system

#clean:
#	rm main

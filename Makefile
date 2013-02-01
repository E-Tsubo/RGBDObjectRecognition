all: pbmrecog

pbmrecog: main.o pbm.o kerneldesc.o detector.o
	g++ -g -fopenmp -o pbmrecog main.o pbm.o kerneldesc.o liblinear/tron.cpp liblinear/linear.cpp liblinear/blas/blas.a \
	`pkg-config opencv matio --cflags --libs` -lhdf5 -IInclude -lboost_filesystem -lboost_system

main.o: main.cpp
	g++ -g -fopenmp -c main.cpp `pkg-config opencv matio --cflags --libs` -lhdf5 -IInclude -lboost_filesystem -lboost_system

pbm.o: pbm.cpp
	g++ -g -fopenmp -c pbm.cpp `pkg-config opencv matio --cflags --libs` -lhdf5 -IInclude -lboost_filesystem -lboost_system

kerneldesc.o: kerneldesc.cpp
	g++ -g -fopenmp -c kerneldesc.cpp `pkg-config opencv matio --cflags --libs` -lhdf5 -IInclude -lboost_filesystem -lboost_system

detector.o: detector.cpp
	g++ -g -c detector.cpp -IInclude `pkg-config opencv pcl_common-1.5 pcl_io-1.5 --cflags --libs`

clean:
	rm pbmrecog main.o pbm.o kerneldesc.o


#main: main.cpp
#	g++ -O3 -fopenmp main.cpp pbm.cpp kerneldesc.cpp liblinear/tron.cpp liblinear/linear.cpp liblinear/blas/blas.a -o main `pkg-config opencv matio --cflags --libs` -lhdf5 -IInclude -lboost_filesystem -lboost_system

#clean:
#	rm main

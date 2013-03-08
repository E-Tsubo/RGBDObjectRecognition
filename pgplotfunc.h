#include <iostream>
#include "cpgplot.h"

void setAxis( int classNum )
{
  //cpgaxis();
  cpgsci(1);
  cpgenv( 1, classNum, -1, 1, 0, 1 );// xmin, xmax, ymin, ymax, zmin(0), zmax(1)
}

void initpg( int classNum )
{
  cpgopen("/xserv");
  cpgpap( 5, 1.0 );//window size, aspect ratio
  cpgask(false);//interactive off
  //cpgenv( 1, 1+classNum, -1, 1, 0, 1 );// xmin, xmax, ymin, ymax, zmin(0), zmax(1)
  setAxis( classNum );
}

void delpg()
{
  cpgclos();
  cpgend();
}

void erasepg( int classNum )
{
  cpgeras();
  setAxis( classNum );
}

void drawpg( int classNum, PBM* pbm, int colorIdx )
{
  //static int colorFlag = 2;
  //cpgsci(colorFlag); colorFlag++;
  //if( colorFlag > 13 ) colorFlag = 2;
  cpgsci( colorIdx + 2 );
  
  for( int i = 1; i < classNum; i++ ){
    if( i == 1 )
      cpgmove( i, pbm->getResult(i-1) );
    else
      cpgdraw( i, pbm->getResult(i-1) );
  }
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <bcm2835.h>

using namespace cv;
using namespace std;

ofstream outfile;
char imname[30];
char outname[30];
int imnum = 0;
int imnummax = 10;
Mat image, h;
double tempfactor = 0.02;
double temp = 0;
char numstr[10];

int irtread() {

  char buf[6];
  char reg;
  reg=7;
  bcm2835_i2c_write(&reg, 1);
  bcm2835_i2c_read_register_rs(&reg, &buf[0],3);
  temp = tempfactor * (double)(buf[0]+(buf[1]<<8));
   
  return 0;
}

int firstimage(){
  char command[100];
  sprintf(numstr,"%04d",imnum);
  strcpy(imname,"../pics/");
  strcat(imname,numstr);
  strcat(imname,".BMP");
  cout << imname << std::endl;
  while(access(imname,F_OK) != -1){
    imnum++;
    sprintf(numstr,"%04d",imnum);
    strcpy(imname,"../pics/");
    strcat(imname,numstr);
    strcat(imname,".BMP");
    cout << imname << std::endl;
  };
  strcpy(command,"raspistill -n -w 640 -h 400 -t 0 -o ");
  strcat(command,imname);
  system(command);
  cout << command << std::endl;
  return 0;
};

int hsegment(){
 
    const int N=2;//classes
    const int d=1;//channels
    const int n=h.rows*h.cols;
    Mat samples(n,d,CV_32F);
    Mat means(N,d,CV_32F), weights(1,N,CV_32F);
    Mat segment3(h.rows,h.cols,CV_8UC3);
    Mat segment1(h.rows,h.cols,CV_8UC1);
    Mat labels(n,1,CV_8U);
    
    Scalar colors[] = {Scalar(0,0,255), Scalar(0,255,0), Scalar(0,255,255), Scalar(255,255,0)};
   
    CvEMParams params;
    CvEM em;

    samples = h.reshape(1,n);

    // initialize model parameters
    params.covs = NULL;
    params.means = NULL;
    params.weights = NULL;
    params.probs = NULL;
    params.nclusters = N;
    params.cov_mat_type = CvEM::COV_MAT_SPHERICAL;
    params.start_step = CvEM::START_AUTO_STEP;
    params.term_crit.max_iter = 300;
    params.term_crit.epsilon = 0.1;
    params.term_crit.type = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
    em.train( samples, Mat(), params, &labels);//run EM 

    segment1 = labels.reshape(1,h.rows);
    for(int i=0;i<segment1.rows;i++){
      for(int j=0;j<segment1.cols;j++){
	Scalar c=colors[segment1.ptr<int>(i)[j]];
	circle(segment3,Point(j,i),1,c,CV_FILLED);
      }
    } 
    means = em.get_means();  
    weights = em.get_weights();
    for(int i=0;i<N;i++){
      for(int j=0;j<d;j++){
	outfile <<  " " <<  means.ptr<double>(i)[j] << " " <<   weights.ptr<double>(0)[i];
      }
    };    

    imwrite(outname,segment3);

    return 0;
}

int sample(){

    Mat hsv;
    vector<Mat> hsvsplit;
    time_t outtime;
    struct tm *ptm;

    image = imread(imname, CV_LOAD_IMAGE_COLOR); // Read the file
    irtread();

    if(! image.data ){
      cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }

    time(&outtime);
    ptm = gmtime(&outtime);

    cvtColor( image, hsv, CV_BGR2HSV );//convert to hsv

    split(hsv, hsvsplit);//split into channels
    h = hsvsplit[0];

    h.convertTo(h, CV_32F);
    h *= 2.0;//opencv h is h/2 to fit in 0-255 range

    outfile << ptm->tm_mon << "/"<< ptm->tm_mday << " " << ptm->tm_hour << ":"<< ptm->tm_min << ":"<< ptm->tm_sec << " " << imname;

    hsegment();//segment image, write EM parameters to file, write segmented image to file

    outfile << " " << temp << std::endl;

    return 0;
}

int main( int argc, char** argv ){

  if(!bcm2835_init()){
    cout<<"Failed bcm2835 init"<<std::endl;
    return -1;
  };

  bcm2835_i2c_begin();
  bcm2835_i2c_set_baudrate(25000);
  bcm2835_i2c_setSlaveAddress(0x5a);  
  
  if (!ifstream("cvirt.dat")){
    outfile.open("cvirt.dat",ios_base::out|ios_base::app);
    outfile << "Time Name Mean1 Frac1 Mean2 Frac2 "  << std::endl; 
  }else{
    outfile.open("cvirt.dat",ios_base::out|ios_base::app);
  };
  
  if(argc<2){
    firstimage();
  }else{
    strcpy(imname,"../pics/");
    strcat(imname,argv[1]);   
  };
  
  strcpy(outname,"../pics/S");
  if(argc<2){
    strcat(outname,numstr);
    strcat(outname,".BMP");
  }else{
    strcat(outname,argv[1]);
  }

  sample();

  if(argc<2){  
    while(imnum<imnummax-1){
      imnum++;   
      char command[100];
      sprintf(numstr,"%04d",imnum);
      strcpy(imname,"../pics/");
      strcat(imname,numstr);
      strcat(imname,".BMP");
      strcpy(command,"raspistill -n -w 640 -h 400 -t 0 -o ");
      strcat(command,imname);
      system(command);
      cout << command << std::endl;
      strcpy(outname,"../pics/S");
      strcat(outname,numstr);
      strcat(outname,".BMP");
      sample();
    }
  }

  outfile.close();
  bcm2835_i2c_end();
  bcm2835_close();

  return 0;

}

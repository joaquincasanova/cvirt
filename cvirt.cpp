#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;

int hvhist(Mat hsv){
    int hbins = 32, vbins = 32;
    int histSize[] = {hbins, vbins};
    float hrange[] = {0, 180};
    float vrange[] = {0, 255};
    const float* ranges[] = { hrange, vrange };
    Mat hist;
    // we compute the histogram from the 0-th and 2-st channels
    int channels[] = {0, 2};

    calcHist( &hsv, 1, channels, Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
    double maxVal=0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);

    int scale = 10;
    Mat histImg = Mat::zeros(vbins*scale, hbins*scale, CV_8UC3);

    for( int hh = 0; hh < hbins; hh++ )
        for( int vv = 0; vv < vbins; vv++ )
        {
            float binVal = hist.at<float>(hh, vv);
            int intensity = cvRound(binVal*255/maxVal);
            rectangle( histImg, Point(hh*scale, vv*scale),
                        Point( (hh+1)*scale - 1, (vv+1)*scale - 1),
                        Scalar::all(intensity),
                        CV_FILLED );
        }

    namedWindow( "H-V Histogram", 1 );
    imshow( "H-V Histogram", histImg );
    waitKey(0);
    destroyWindow( "H-V Histogram" );
    return 0;
}

int hsegment(Mat h, CvEM emx){

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
    cout <<  "samples:  " << samples.rows <<  " dimensions:  " <<  samples.cols <<  std::endl;

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
    means = em.get_means();  
    weights = em.get_weights();
    for(int i=0;i<N;i++){
      for(int j=0;j<d;j++){
	cout << i << " " << j <<  " " <<  means.ptr<double>(i)[j] << " " <<   weights.ptr<double>(0)[i] <<  std::endl;
      }
    };    
    segment1 = labels.reshape(1,h.rows);
    for(int i=0;i<segment1.rows;i++){
      for(int j=0;j<segment1.cols;j++){
	Scalar c=colors[segment1.ptr<int>(i)[j]];
	circle(segment3,Point(j,i),1,c,CV_FILLED);
      }
    } 
    namedWindow( "Segment", CV_WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Segment", segment3);                // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    destroyWindow( "Segment" );
    
    return 0;
}

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: cvirt ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image, hsv, h;
    vector<Mat> hsvsplit;
    CvEM em;
    clock_t timer1, timer2;
    double seconds;

    /*char command[100];
    strcpy(command,"raspistill -n -w 640 -h 400 -t 0 -o ");
    strcat(command,argv[1]);
    system(command);*/

    image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file

    if(! image.data )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Original", CV_WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Original", image );                // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    destroyWindow( "Original" );

    timer1 = clock();
    cvtColor( image, hsv, CV_BGR2HSV );//convert to hsv
    timer2 = clock();
    seconds = (double)(timer2-timer1)/CLOCKS_PER_SEC;
    cout << seconds << " for cvtColor" << std::endl;

    split(hsv, hsvsplit);//split into channels
    h = hsvsplit[0];

    namedWindow( "Hue", CV_WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Hue", h);                // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    destroyWindow( "Hue" );

    h.convertTo(h, CV_32F);
    h *= 2.0;//opencv h is h/2 to fit in 0-255 range
    hvhist(hsv);

    timer1 = clock();
    hsegment(h, em);
    timer2 = clock();
    seconds = (double)(timer2-timer1)/CLOCKS_PER_SEC;
    cout << seconds << " for EM" << std::endl;

    return 0;

}

#include <bits/stdc++.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

Mat input_img, image, inpaintMask;
Point prevPt(-1,-1);
int thickness = 5;

class Inpainter
{
public:
    const static int DEFAULT_HALF_PATCH_WIDTH=3;
    const static int MODE_ADDITION=0;
    const static int MODE_MULTIPLICATION=1;
    const static int ERROR_INPUT_MAT_INVALID_TYPE=0;
    const static int ERROR_INPUT_MASK_INVALID_TYPE=1;
    const static int ERROR_MASK_INPUT_SIZE_MISMATCH=2;
    const static int ERROR_HALF_PATCH_WIDTH_ZERO=3;
    const static int CHECK_VALID=4;

    Inpainter(cv::Mat inputImage,cv::Mat mask,int halfPatchWidth=4,int mode=1);

    cv::Mat inputImage;
    cv::Mat mask,updatedMask;
    cv::Mat result;
    cv::Mat workImage;
    cv::Mat sourceRegion;
    cv::Mat targetRegion;
    cv::Mat originalSourceRegion;
    cv::Mat gradientX;
    cv::Mat gradientY;
    cv::Mat confidence;
    cv::Mat data;
    cv::Mat LAPLACIAN_KERNEL,NORMAL_KERNELX,NORMAL_KERNELY;
    cv::Point2i bestMatchUpperLeft,bestMatchLowerRight;
    std::vector<cv::Point> fillFront;
    std::vector<cv::Point2f> normals;
    int mode;
    int halfPatchWidth;
    int targetIndex;

    int checkValidInputs();
    void calculateGradients();
    void initializeMats();
    void computeFillFront();
    void computeConfidence();
    void computeData();
    void computeTarget();
    void computeBestPatch();
    void updateMats();
    bool checkEnd();
    void getPatch(cv::Point2i &centerPixel, cv::Point2i &upperLeft, cv::Point2i &lowerRight);
    void inpaint();
};

static void onMouse(int event, int x, int y, int flags, void*){
	if(event == EVENT_LBUTTONUP || flags != EVENT_FLAG_LBUTTON){
		prevPt = Point(-1,-1);
	}
	else if (event == EVENT_LBUTTONDOWN)
		prevPt = Point(x,y);

	else if(event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_LBUTTON){
		Point pt(x,y);
		if(prevPt.x < 0)
			prevPt = pt;

		line(inpaintMask, prevPt, pt, Scalar::all(255), thickness,8,0);
		line(image, prevPt, pt, Scalar::all(255), thickness,8,0);
		prevPt = pt;
		imshow("Input Image", image);
	}

}

int main(int argc, char *argv[]){

	// First argument is Image path
	// Second argument is halfPatchWidth

	//int halfPatchWidth = 4;

	if(argc < 3){
		cout<<"Parameter missing"<<endl;
		return 0;
	}

	input_img = imread(argv[1],CV_LOAD_IMAGE_COLOR);

	if(!input_img.data){
		cout<<"Unable to open input image"<<endl;
		return 0;
	}

	image = input_img.clone();

	inpaintMask = Mat::zeroes(image.size(), CV_8U);

	namedWindow("Input Image", WINDOW_AUTOSIZE);
	imshow("Input Image", image);
	setMouseCallback("image", onMouse, NULL);

	while(1){
		char c = waitKey();

		if(c == 'e')
			break;

		if(c == 'r'){

		}
		if(c == 'i' || c == ' '){

		}

		if(c == 's'){
			thickness++;
			cout<<"Thickness =  "<<thickness<<endl;
		}

		if(c == 'a'){
			thickness--;
			cout<<"Thickness = "<<thickness<<endl;
		}

		if(thickness<3)
			thickness = 3;
		if(thickness > 12)
			thickness = 12;
	}
}


#include <bits/stdc++.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

Mat originalImage, image, inpaintMask;
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

    Inpainter(Mat inputImage,Mat mask,int halfPatchWidth=4,int mode=1)
    {
    	this->inputImage=inputImage.clone();
		this->mask=mask.clone();
		this->updatedMask=mask.clone();
		this->workImage=inputImage.clone();
		this->result.create(inputImage.size(),inputImage.type());
		this->mode=mode;
		this->halfPatchWidth=halfPatchWidth;
    }

    Mat inputImage;
    Mat mask,updatedMask;
    Mat result;
    Mat workImage;
    Mat sourceRegion;
    Mat targetRegion;
    Mat originalSourceRegion;
    Mat gradientX;
    Mat gradientY;
    Mat confidence;
    Mat data;
    Mat LAPLACIAN_KERNEL,NORMAL_KERNELX,NORMAL_KERNELY;
    Point2i bestMatchUpperLeft,bestMatchLowerRight;
    vector<Point> fillFront;
    vector<Point2f> normals;
    int mode;
    int halfPatchWidth;
    int targetIndex;


    int checkValidInputs()
    {
    	if(this->inputImage.type()!=CV_8UC3)
    	    return 0;//ERROR_INPUT_MAT_INVALID_TYPE
    	if(this->mask.type()!=CV_8UC1)
    	    return 1;//ERROR_INPUT_MASK_INVALID_TYPE;
    	if(!CV_ARE_SIZES_EQ(&mask,&inputImage))
    	    return 2;//ERROR_MASK_INPUT_SIZE_MISMATCH;
    	if(halfPatchWidth==0)
    	    return 3;//ERROR_HALF_PATCH_WIDTH_ZERO;
    	return 4;//CHECK_VALID;
    }
    void calculateGradients()
    {
    	Mat srcGray;
	    cvtColor(workImage,srcGray,CV_BGR2GRAY);

	    Scharr(srcGray,gradientX,CV_16S,1,0);
	    convertScaleAbs(gradientX,gradientX);
	    gradientX.convertTo(gradientX,CV_32F);


	    Scharr(srcGray,gradientY,CV_16S,0,1);
	    convertScaleAbs(gradientY,gradientY);
	    gradientY.convertTo(gradientY,CV_32F);






	    for(int x=0;x<sourceRegion.cols;x++){
	        for(int y=0;y<sourceRegion.rows;y++){

	            if(sourceRegion.at<uchar>(y,x)==0){
	                gradientX.at<float>(y,x)=0;
	                gradientY.at<float>(y,x)=0;
	            }/*else
	            {
	                if(gradientX.at<float>(y,x)<255)
	                    gradientX.at<float>(y,x)=0;
	                if(gradientY.at<float>(y,x)<255)
	                    gradientY.at<float>(y,x)=0;
	            }*/

	        }
	    }
	    gradientX/=255;
	    gradientY/=255;
    }
    void initializeMats();
    void computeFillFront();
    void computeConfidence();
    void computeData();

    void computeTarget();
    void computeBestPatch();
    void updateMats();
    bool checkEnd();
    void getPatch(Point2i &centerPixel, Point2i &upperLeft, Point2i &lowerRight);
    
    void inpaint(){
    	namedWindow("updatedMask");
    	namedWindow("inpaint");
    	namedWindow("gradientX");
    	namedWindow("gradientY");
    	initializeMats();
    	calculateGradients();
    	bool notFilled=true;
    	while(notFilled){
    		computeFillFront();
    		computeConfidence();
    		computeData();
    		computeTarget();
    		computeBestPatch();
    		updateMats();
    		notFilled=checkEnd();

    		imshow("updatedMask",updatedMask);
    		imshow("inpaint",workImage);
    		imshow("gradientX",gradientX);
    		imshow("gradientY",gradientY);
    		waitKey(2);
    	}
    	result=workImage.clone();

    	namedWindow("confidence");
    	imshow("confidence",confidence);
    }
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

    int halfPatchWidth = 4;

    if(argc < 3){
        cout<<"Parameter missing"<<endl;
        return 0;
    }

    originalImage = imread(argv[1],CV_LOAD_IMAGE_COLOR);

    if(!originalImage.data){
        cout<<"Unable to open input image"<<endl;
        return 0;
    }

    image = originalImage.clone();

    inpaintMask = Mat::zeroes(image.size(), CV_8U);

    namedWindow("Input Image", WINDOW_AUTOSIZE);
    imshow("Input Image", image);
    setMouseCallback("image", onMouse, NULL);

    while(1){
        char c = waitKey();

        if(c == 'e')
            break;

        if(c == 'i'){
            Inpainter i(originalImage, inpaintMask, halfPatchWidth);
            if(i.checkValidInputs() == i.CHECK_VALID){
                i.inpaint();
                imwrite("Result.jpg", i.result);
                inpaintMask = Scalar::all(0);
                namedWindow("Result");
                imshow("Result",i.result);
            }
            else{
                cout<<"Invalid Parameters"<<endl;
            }
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

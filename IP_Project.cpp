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
    //edge detection using sobel derivative
    void calculateGradients()
    {
    	Mat srcGray;
    	//convert image from one color space to other
	    cvtColor(workImage,srcGray,CV_BGR2GRAY);
	    //Sobel Derivatives for x 
	    Scharr(srcGray,gradientX,CV_16S,1,0);
	    //scale and convert to 8 bits
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
	            }
	        }
	    }
	    gradientX/=255;
	    gradientY/=255;
    }
    void initializeMats()
    {
    	//threshold mask image into confidence image
    	threshold(this->mask,this->confidence,10,255,CV_THRESH_BINARY);
    	//thresholding operations
	    threshold(confidence,confidence,2,1,CV_THRESH_BINARY_INV);
	    confidence.convertTo(confidence,CV_32F);
	    
	    this->sourceRegion=confidence.clone();
	    this->sourceRegion.convertTo(sourceRegion,CV_8U);
	    this->originalSourceRegion=sourceRegion.clone();

	    threshold(mask,this->targetRegion,10,255,CV_THRESH_BINARY);
	    threshold(targetRegion,targetRegion,2,1,CV_THRESH_BINARY);
	    targetRegion.convertTo(targetRegion,CV_8U);
	    data=Mat(inputImage.rows,inputImage.cols,CV_32F,Scalar::all(0));


	    LAPLACIAN_KERNEL=Mat::ones(3,3,CV_32F);
	    LAPLACIAN_KERNEL.at<float>(1,1)=-8;
	    NORMAL_KERNELX=Mat::zeros(3,3,CV_32F);
	    NORMAL_KERNELX.at<float>(1,0)=-1;
	    NORMAL_KERNELX.at<float>(1,2)=1;
	    transpose(NORMAL_KERNELX,NORMAL_KERNELY);
    }
    void computeFillFront();
    void computeConfidence(){
        Point2i a,b;    // Integer points

        for(int i=0;i<fillFront.size();i++){
            Point2i currentPoint = fillFront.at(i);
            getPatch(currentPoint, a, b);
            float total = 0;

            for(int x1 = a.x; x1<=b.x; x1++){
                for(int y1 = a.y; y1<=b.y; y1++){
                    if(targetRegion.at<uchar>(y1,x1) == 0){
                        total+=confidence.at<float>(y1,x1);
                    }
                }
            }
            confidence.at<float>(currentPoint.y, currentPoint.x) = total/((b.x-a.x+1)*(b.y-a.y+1));
        }
    }
    void computeData();

    void computeTarget();
    void computeBestPatch();

    //It updates the workImage and gradient Images with the values as present in the patch
    void updateMats(){
    	Point2i targetPoint=fillFront.at(targetIndex);
    	Point2i a,b;
    	getPatch(targetPoint,a,b);
    	int width=b.x-a.x+1;
    	int height=b.y-a.y+1;

    	for(int x=0;x<width;x++){
    		for(int y=0;y<height;y++){
    			if(sourceRegion.at<uchar>(a.y+y,a.x+x)==0){
					workImage.at<Vec3b>(a.y+y,a.x+x)=workImage.at<Vec3b>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
					gradientX.at<float>(a.y+y,a.x+x)=gradientX.at<float>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
					gradientY.at<float>(a.y+y,a.x+x)=gradientY.at<float>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
					confidence.at<float>(a.y+y,a.x+x)=confidence.at<float>(targetPoint.y,targetPoint.x);
					sourceRegion.at<uchar>(a.y+y,a.x+x)=1;
					targetRegion.at<uchar>(a.y+y,a.x+x)=0;
					updatedMask.at<uchar>(a.y+y,a.x+x)=0;
				}
    		}
    	}
    }

    bool checkEnd(){
    	for(int x=0;x<sourceRegion.cols;x++){
    		for(int y=0;y<sourceRegion.rows;y++){
    			if(sourceRegion.at<uchar>(y,x)==0){
    				return true;
    			}
    		}
    	}
    	return false;
    }

    void getPatch(Point2i &centerPixel, Point2i &upperLeft, Point2i &lowerRight){
    	int x,y;
    	x=centerPixel.x;
    	y=centerPixel.y;

    	int minX=max(x-halfPatchWidth,0);
    	int maxX=min(x+halfPatchWidth,workImage.cols-1);
    	int minY=max(y-halfPatchWidth,0);
    }
    
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

    void computeTarget(){
    	targetIndex = 0;
    	float maxPrior = 0;
    	float prior = 0;
    	Point2i currentPoint;
    	for(int i=0;i<fillFront.size();i++){
    		currentPoint = fillFront[i];
    		prior = data.at<float>(currentPoint.y,currentPoint.x)*confidence.at<float>(currentPoint.y,currentPoint.x);
    		if(prior>maxPrior){
    			maxPrior = prior;
    			targetIndex = i;
    		}
    	}
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

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

    void computeFillFront(){
    	printf("ENTERING FUNCTION computeFillFront\n");
        Mat sourceGradientX,sourceGradientY,boundryMat;
        // filter2D --> Convolves an image with the kernel. 
        // TargetRegion -> i/p image
        // Laplacian Kernel -> i/p kernel
        // boundryMat --> o/p image
        filter2D(targetRegion,boundryMat,CV_32F,LAPLACIAN_KERNEL);
        filter2D(sourceRegion,sourceGradientX,CV_32F,NORMAL_KERNELX);
        filter2D(sourceRegion,sourceGradientY,CV_32F,NORMAL_KERNELY);

        fillFront.clear();
        normals.clear();

        for(int x=0; x < boundryMat.cols;x++){
            for(int y=0;y<boundryMat.rows;y++){

                if(boundryMat.at<float>(y,x)>0){
                    fillFront.push_back(Point2i(x,y));

                    float dx=sourceGradientX.at<float>(y,x);
                    float dy=sourceGradientY.at<float>(y,x);
                    Point2f normal(dy,-dx);
                    float tempF=sqrt((normal.x*normal.x)+(normal.y*normal.y));
                    if(tempF!=0){

                    normal.x=normal.x/tempF;
                    normal.y=normal.y/tempF;

                    }
                    normals.push_back(normal);

                }
            }
        }
    }
    void computeConfidence(){
    	printf("ENTERING FUNCTION computeConfidence\n");
        Point2i a,b;    // Integer points

        // fillfront (vector) 
        for(int i=0;i<fillFront.size();i++){
            Point2i currentPoint = fillFront.at(i);
            getPatch(currentPoint, a, b);   // 
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

    void computeData(){
    	printf("ENTERING FUNCTION computeData\n");
    	for(int i=0;i<fillFront.size();i++){
	        cv::Point2i currentPoint=fillFront.at(i);
	        cv::Point2i currentNormal=normals.at(i);
	        data.at<float>(currentPoint.y,currentPoint.x)=std::fabs(gradientX.at<float>(currentPoint.y,currentPoint.x)*currentNormal.x+gradientY.at<float>(currentPoint.y,currentPoint.x)*currentNormal.y)+.001;
	    }
    }

    void computeBestPatch(){
        printf("ENTERING FUNCTION computeBestPatch\n");
	    double minError=9999999999999999,bestPatchVarience=9999999999999999;
	    cv::Point2i a,b;
	    cv::Point2i currentPoint=fillFront.at(targetIndex);
	    cv::Vec3b sourcePixel,targetPixel;
	    double meanR,meanG,meanB;
	    double difference,patchError;
	    bool skipPatch;
	    getPatch(currentPoint,a,b);

	    int width=b.x-a.x+1;
	    int height=b.y-a.y+1;
	    for(int x=0;x<=workImage.cols-width;x++){
	        for(int y=0;y<=workImage.rows-height;y++){
	            patchError=0;
	            meanR=0;meanG=0;meanB=0;
	            skipPatch=false;

	            for(int x2=0;x2<width;x2++){
	                for(int y2=0;y2<height;y2++){
	                    if(originalSourceRegion.at<uchar>(y+y2,x+x2)==0){
	                        skipPatch=true;
	                        break;
	                     }

	                    if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0)
	                        continue;

	                    sourcePixel=workImage.at<cv::Vec3b>(y+y2,x+x2);
	                    targetPixel=workImage.at<cv::Vec3b>(a.y+y2,a.x+x2);

	                    for(int i=0;i<3;i++){
	                        difference=sourcePixel[i]-targetPixel[i];
	                        patchError+=difference*difference;
	                    }
	                    meanB+=sourcePixel[0];meanG+=sourcePixel[1];meanR+=sourcePixel[2];
	                }
	                if(skipPatch)
	                    break;
	            }

	            if(skipPatch)
	                continue;
	            if(patchError<minError){
	                minError=patchError;
	                bestMatchUpperLeft=cv::Point2i(x,y);
	                bestMatchLowerRight=cv::Point2i(x+width-1,y+height-1);

	                double patchVarience=0;
	                for(int x2=0;x2<width;x2++){
	                    for(int y2=0;y2<height;y2++){
	                        if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0){
	                            sourcePixel=workImage.at<cv::Vec3b>(y+y2,x+x2);
	                            difference=sourcePixel[0]-meanB;
	                            patchVarience+=difference*difference;
	                            difference=sourcePixel[1]-meanG;
	                            patchVarience+=difference*difference;
	                            difference=sourcePixel[2]-meanR;
	                            patchVarience+=difference*difference;
	                        }

	                    }
	                }
	                bestPatchVarience=patchVarience;

	            }else if(patchError==minError){
	                double patchVarience=0;
	                for(int x2=0;x2<width;x2++){
	                    for(int y2=0;y2<height;y2++){
	                        if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0){
	                            sourcePixel=workImage.at<cv::Vec3b>(y+y2,x+x2);
	                            difference=sourcePixel[0]-meanB;
	                            patchVarience+=difference*difference;
	                            difference=sourcePixel[1]-meanG;
	                            patchVarience+=difference*difference;
	                            difference=sourcePixel[2]-meanR;
	                            patchVarience+=difference*difference;
	                        }

	                    }
	                }
	                if(patchVarience<bestPatchVarience){
	                    minError=patchError;
	                    bestMatchUpperLeft=cv::Point2i(x,y);
	                    bestMatchLowerRight=cv::Point2i(x+width-1,y+height-1);
	                    bestPatchVarience=patchVarience;
	                }
	            }
	        }
	    }
	}

    //It updates the workImage and gradient Images with the values as present in the patch
    void updateMats(){
        printf("ENTERING FUNCTION updateMats\n");
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
        printf("ENTERING FUNCTION checkEnd\n");
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

	    int minX=std::max(x-halfPatchWidth,0);
	    int maxX=std::min(x+halfPatchWidth,workImage.cols-1);
	    int minY=std::max(y-halfPatchWidth,0);
	    int maxY=std::min(y+halfPatchWidth,workImage.rows-1);


	    upperLeft.x=minX;
	    upperLeft.y=minY;

	    lowerRight.x=maxX;
	    lowerRight.y=maxY;
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
        printf("ENTERING FUNCTION computeTarget\n");
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

static void onMouse( int event, int x, int y, int flags, void* )
{
    if(event == cv::EVENT_LBUTTONUP||!(flags & cv::EVENT_FLAG_LBUTTON) )
        prevPt = cv::Point(-1,-1);
    else if( event == cv::EVENT_LBUTTONDOWN )
        prevPt = cv::Point(x,y);
    else if( event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON) )
    {
        cv::Point pt(x,y);
        if( prevPt.x < 0 )
            prevPt = pt;
        cv::line( inpaintMask, prevPt, pt, cv::Scalar::all(255), thickness, 8, 0 );
        cv::line( image, prevPt, pt, cv::Scalar::all(255), thickness, 8, 0 );
        prevPt = pt;
        cv::imshow("image", image);
    }
}

int main(int argc, char *argv[])
{

    //we expect three arguments.
    //the first is the image path.
    //the second is the mask path.
    //the third argument is the halfPatchWidth

    //in case halPatchWidth is not specified we use a default value of 3.
    //in case only image path is speciifed, we use manual marking of mask over the image.
    //in case image name is also not specified , we use default image default.jpg.


    int halfPatchWidth=4;

    if(argc>=4)
    {
        std::stringstream ss;
        ss<<argv[3];
        ss>>halfPatchWidth;
    }

    char* imageName = argc >= 2 ? argv[1] : (char*)"default.jpg";

    originalImage=cv::imread(imageName,CV_LOAD_IMAGE_COLOR);

    if(!originalImage.data){
        std::cout<<std::endl<<"Error unable to open input image"<<std::endl;
        return 0;
    }

    image=originalImage.clone();



    bool maskSpecified=false;
    char* maskName;
    if(argc >= 3){
       maskName=argv[2];
       maskSpecified=true;
    }

    if(maskSpecified){
        inpaintMask=cv::imread(maskName,CV_LOAD_IMAGE_GRAYSCALE);
        Inpainter i(originalImage,inpaintMask,halfPatchWidth);
        if(i.checkValidInputs()==i.CHECK_VALID){
            i.inpaint();
            cv::imwrite("result.jpg",i.result);
            cv::namedWindow("result");
            cv::imshow("result",i.result);
            cv::waitKey();
        }else{
            std::cout<<std::endl<<"Error : invalid parameters"<<std::endl;
        }
    }
    else
    {
        std::cout<<std::endl<<"mask not specified , mark manually on input image"<<std::endl;
        inpaintMask = cv::Mat::zeros(image.size(), CV_8U);
        cv::namedWindow( "image", 1 );
        cv::imshow("image", image);
        cv::setMouseCallback( "image", onMouse, 0 );

        for(;;)
            {
                char c = (char)cv::waitKey();

                if( c == 'e' )
                    break;

                if( c == 'r' )
                {
                    inpaintMask = cv::Scalar::all(0);
                    image=originalImage.clone();
                    cv::imshow("image", image);
                }

                if( c == 'i' || c == ' ' )
                {
                    Inpainter i(originalImage,inpaintMask,halfPatchWidth);
                    if(i.checkValidInputs()==i.CHECK_VALID){
                        i.inpaint();
                        cv::imwrite("result.jpg",i.result);
                        inpaintMask = cv::Scalar::all(0);
                        cv::namedWindow("result");
                        cv::imshow("result",i.result);
                    }else{
                        std::cout<<std::endl<<"Error : invalid parameters"<<std::endl;
                    }


                }
                if(c=='s'){
                    thickness++;
                    std::cout<<std::endl<<"Thickness = "<<thickness;
                }
                if(c=='a'){
                    thickness--;
                    std::cout<<std::endl<<"Thickness = "<<thickness;
                }
                if(thickness<3)
                    thickness=3;
                if(thickness>12)
                    thickness=12;
            }

    }
    return 0;
}
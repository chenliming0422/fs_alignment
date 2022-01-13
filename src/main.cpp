////////////////////////////////////////////////////////////////////////////////////////
//
// This code is for ICIA algorithm testing
// 
// Copyright(c) 2022 XYZT Lab, Purdue University.
// Author: Liming Chen
// Date: 2022-01-04
// Contact: chen3496@purdue.edu 
// 
///////////////////////////////////////////////////////////////////////////////////////

#include <direct.h>
#include <io.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>

//OpenCV library
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//Project
#include "CImageFilters.h"
#include "CPhaseWrapUnwrap.h"
#include "image_alignment.h"

using namespace std;
using namespace cv;

int main()
{
	string imagePath = "../../data/test/screw.png";

	Mat image = imread(imagePath, IMREAD_GRAYSCALE);
	imshow("image", image);
	waitKey();
	destroyWindow("image");

	Mat test_warp = 0.01 * Mat::eye(2, 3, CV_64FC1);
	test_warp.at<double>(0, 2) = 2;
	test_warp.at<double>(1, 2) = 3;

	test(image, test_warp);

	return 0;
}
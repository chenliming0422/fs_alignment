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
	int stackSize = 13;
	string stackPath = "../../data/Screw";
	string outputPath = "../../data/Screw/align";
	vector<Mat> imageStack;
	vector<Mat> warpStack;

	for (int i = 0; i < stackSize; i++)
	{
		string imageName = stackPath + "/" + to_string(i) + ".png";
		Mat image = imread(imageName, IMREAD_GRAYSCALE);
		imageStack.push_back(image.clone());
	}

	align_image_stack(imageStack, &warpStack);
	
	/*
	for (int i = 1; i < stackSize; i++)
	{
		Mat warp_image;
		warp_affine(imageStack[i], warpStack[i - 1], &warp_image);
		string name = outputPath + "/" + to_string(i) + ".png";
		imwrite(name, warp_image);
	}*/

	return 0;
}
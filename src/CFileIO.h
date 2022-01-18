#ifndef __C_PHASE_MAP_FILE__H
#define __C_PHASE_MAP_FILE__H

#include <vector>

#include <fstream>
#include <iostream>
#include <iomanip>

//OpenCV library
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;

class CFileIO
{
public:
	void WritePhaseMapFile(const char * nFileName, float * nPhaseData, int nImageWidth, int nImageHeight);

	void Write3DPoints(const char * nFileName, const cv::Mat & nMatrix);

	void WritePointVector3D(const char * nFileName, const vector<cv::Point3f> & nPoints);
	void WritePointVector2D(const char * nFileName, const vector<cv::Point2f> & nPoints);

	void WriteMatrix(const char * nFileName, const cv::Mat & nMatrix);

	void WriteImageFile(const char* nFileName, const cv::Mat& nImage);

	void WriteArray(const char* nFileName, const int * nArray, int size);
};


#endif

#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include "CImageFilters.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#pragma comment(lib, "opencv_world420.lib")
using namespace std;
using namespace cv;

class CPhaseWrapUnwrap
{
public:
	CPhaseWrapUnwrap(int width, int height);
public:
	void computePhase(float* wrapphase, unsigned char* fringeImage[], const int nStep, float* Iavg = nullptr, float *amplitude = nullptr);
	void unwrapThreeFreqPhaseLine(float* unwrappedPhase, unsigned char* fringeImage[],
		int fringePeriods[], int fringeSteps[], int spikeFilterSize, float* Iavg = nullptr, float* amplitude = nullptr);
	void unwrapThreeFreqPhaseLine(float* unwrappedPhase, unsigned char** fringeImage,
		vector<int> fringePeriods, vector<int>fringeSteps, int spikeFilterSize, float* Iavg = nullptr, float* amplitude = nullptr);
	int m_cameraWidth, m_cameraHeight, m_cameraSize;
};


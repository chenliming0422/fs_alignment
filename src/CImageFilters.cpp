#include "CImageFilters.h"
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

CImageFilters::CImageFilters(int width, int height) :
	m_imageWidth(width), m_imageHeight(height), m_imageSize(width* height)
{
}

CImageFilters::~CImageFilters(void)
{
}

void CImageFilters::medianPhaseFilter(float* imageIn, float* imageOut, int filterSize)
{
	if (filterSize < 2) return;

	memcpy(imageOut, imageIn, sizeof(imageIn[0]) * m_imageSize);
	
	int hsize = (filterSize) / 2;
	vector<float> vect;
	// Median filter of the phase image
	for (int i = hsize; i < m_imageHeight - hsize; i++)
	{
		for (int j = hsize; j < m_imageWidth - hsize; j++)
		{
			int idx = i * m_imageWidth + j;
			vect.clear();
			for (int m = -hsize; m <= hsize; m++)
			{
				for (int n = -hsize; n <= hsize; n++)
				{
					int idx1 = (i + m) * m_imageWidth + (j + n);
					vect.push_back(imageIn[idx1]);
				}
			}
			sort(vect.begin(), vect.end());
			imageOut[idx] = vect[vect.size() / 2];
		}
	}
}

//--------------------------------------------------------------------
// Compute 2D squared Gaussian filter
//
// Input:
//		filterSize	=  size of the filter
//
// Return:
//		filter		= data for generated filter 
//		
//
// Source code was finalized by Professor Song Zhang at Purdue University
// Contact: szhang15@purdue.edu
// Date: 08 / 17 / 2015
//--------------------------------------------------------------------
void CImageFilters::ComputeGaussainFilter2D(float*& filter, int filterSize)
{
	float thegma2 = (float)(2 * (filterSize / 3.0) * (filterSize / 3.0));
	if (filter != NULL) delete[] filter;
	int hsize = (filterSize >> 1);
	filter = new float[filterSize * filterSize];

	float sum = 0.0f;
	for (int i = 0; i < filterSize; i++)
	{
		for (int j = 0; j < filterSize; j++)
		{
			filter[i * filterSize + j] = exp((-(i - hsize) * (i - hsize) - (j - hsize) * (j - hsize)) / thegma2);
			sum += filter[i * filterSize + j];
		}
	}
	sum = 1.0f / sum;
	for (int i = 0; i < filterSize * filterSize; i++)
	{
		filter[i] *= sum;
	}
}

//--------------------------------------------------------------------
//Smooth 8-bit image with Gaussian filter
//
// Input:
//		imageData	=  input image data
//		filterSize	=  size of the filter
//
// Return:
//		imageData	= output smoothed image data  
//		
//
// Source code was finalized by Professor Song Zhang at Purdue University
// Contact: szhang15@purdue.edu
// Date: 08 / 17 / 2015
//--------------------------------------------------------------------
void CImageFilters::GaussianFilter(unsigned char* imageData, int filterSize)
{
	if (filterSize < 3) return; // don't need smooth;
	int s1 = filterSize;
	int i, j, k1, m, n, k;
	int s = filterSize / 2;
	float* F = NULL;
	ComputeGaussainFilter2D(F, s * 2 + 1);
	unsigned char* imageInput = new unsigned char[m_imageSize];
	memcpy(imageInput, imageData, sizeof(imageInput[0]) * m_imageSize);

	for (i = s; i < m_imageHeight - s; i++)
	{
		int lineNo = i * m_imageWidth;
		for (j = s; j < m_imageWidth - s; j++)
		{
			k1 = lineNo + j;

			float temp = 0.0f;
			for (m = -s; m <= s; m++)
			{
				for (n = -s; n <= s; n++)
				{
					k = (i + m) * m_imageWidth + j + n;
					temp += imageInput[k] * F[(m + s) * filterSize + s + n];
				}
			}
			imageData[k1] = (int)(temp + 0.5);
		}
	}

	delete[] imageInput;
	delete[] F;
}

//--------------------------------------------------------------------
//Smooth floating point image with Gaussian filter
//
// Input:
//		imageData	=  input image data
//		filterSize	=  size of the filter
//		mask		=  indication of valid point (none zeros)
//
// Return:
//		imageData	=  output smoothed image data 
//		
//
// Source code was finalized by Professor Song Zhang at Purdue University
// Contact: szhang15@purdue.edu
// Date: 08 / 17 / 2015
//--------------------------------------------------------------------
void CImageFilters::GaussianFilterPhase(float* imageData, unsigned char* mask, int filterSize)
{
	if (filterSize < 3) return; // don't need smooth;
	int s1 = filterSize;
	int i, j, k1, m, n, k;
	int s = filterSize / 2;
	float* F = NULL;
	ComputeGaussainFilter2D(F, s * 2 + 1);
	float* imageInput = new float[m_imageSize];
	memcpy(imageInput, imageData, sizeof(imageInput[0]) * m_imageSize);

	for (i = s; i < m_imageHeight - s; i++)
	{
		int lineNo = i * m_imageWidth;
		for (j = s; j < m_imageWidth - s; j++)
		{
			k1 = lineNo + j;

			if (mask[k1])
			{
				float temp = 0.0f;
				float sumFilter = 0.0f;
				for (m = -s; m <= s; m++)
				{
					for (n = -s; n <= s; n++)
					{
						k = (i + m) * m_imageWidth + j + n;
						if (mask[k])
						{
							temp += imageInput[k] * F[(m + s) * filterSize + s + n];
							sumFilter += F[(m + s) * filterSize + s + n];
						}
					}
				}
				imageData[k1] = temp / sumFilter;
			}
		}
	}

	delete[] imageInput;
	delete[] F;
}

//--------------------------------------------------------------------
// Remove spiky noise on the phase image
// Appy median filter to determine the spiky points, and then determine 
// number of 2pi to be added for those spiky points. 
// NOTE: the spiky noise are caused by incorrect phase unwrapping
//
// Input:
//		Phase		=  input phase data
//		size		=  size of the filter
//		Mask		=  indication of valid point (none zeros)
//
// Return:
//		Phase		=  output of phase image with reduced spikes 
//		
//
// Source code was finalized by Professor Song Zhang at Purdue University
// Contact: szhang15@purdue.edu
// Date: 08 / 17 / 2015
//--------------------------------------------------------------------
void CImageFilters::RemovePhaseSpike(float* Phase, unsigned char* Mask, const int size)
{
	vector<float> vect;
	float two_pi = (float)atan2(1.0, 0.0) * 4.0f;

	float* tmpPhase = new float[m_imageSize];
	memcpy(tmpPhase, Phase, sizeof(tmpPhase[0]) * m_imageSize);
	medianPhaseFilter(Phase, tmpPhase, size);

	// find spiky points and remove them
	for (int i = 0; i < m_imageHeight; i++)
	{
		for (int j = 0; j < m_imageWidth; j++)
		{
			int idx = i * m_imageWidth + j;
			if (Mask[idx])
			{
				float t = (tmpPhase[idx] - Phase[idx]) / (float)(two_pi);
				//int n = int(t+0.5);
				int n = t < 0 ? (int)(t - 0.5) : (int)(t + 0.5);
				Phase[idx] += n * (float)two_pi;
			}
		}
	}

	delete[] tmpPhase;
}

void CImageFilters::removeNormalizedPhaseSpike(float* Phase, unsigned char* Mask, const int size)
{
	vector<float> vect;
	float* tmpPhase = new float[m_imageSize];
	memcpy(tmpPhase, Phase, sizeof(tmpPhase[0]) * m_imageSize);
	medianPhaseFilter(Phase, tmpPhase, size);

	// find spiky points and remove them
	for (int i = 0; i < m_imageHeight; i++)
	{
		for (int j = 0; j < m_imageWidth; j++)
		{
			int idx = i * m_imageWidth + j;
			if (Mask[idx])
			{
				float t = (tmpPhase[idx] - Phase[idx]);
				//int n = int(t+0.5);
				int n = t < 0 ? (int)(t - 0.5) : (int)(t + 0.5);
				Phase[idx] += n;
			}
		}
	}

	delete[] tmpPhase;
}
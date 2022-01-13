#pragma once
class CImageFilters
{
public:
	CImageFilters(int width, int height);
	~CImageFilters(void);

	void GaussianFilter(unsigned char* imageData, int filterSize);
	void GaussianFilterPhase(float* imageData, unsigned char* mask, int filterSize);
	void medianPhaseFilter(float* imageIn, float* imageOut, int filterSize);
	void RemovePhaseSpike(float* Phase, unsigned char* Mask, const int size);
	void removeNormalizedPhaseSpike(float* Phase, unsigned char* Mask, const int size);

private:
	void ComputeGaussainFilter2D(float*& filter, int filterSize);

public:
	int m_imageWidth, m_imageHeight, m_imageSize;
};


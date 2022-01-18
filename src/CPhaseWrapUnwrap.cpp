#include "CPhaseWrapUnwrap.h"


CPhaseWrapUnwrap::CPhaseWrapUnwrap(int width, int height)
{
	m_cameraWidth = width;
	m_cameraHeight = height;
	m_cameraSize = width * height;
}

void CPhaseWrapUnwrap::computePhase(float* wrapphase, unsigned char* fringeImage[], const int nStep, float *Iavg, float *amplitude)
{
	float pi = atan2(1.0f, 0.0f) * 2.0f;
	float two_pi = 2.0f * pi;
	float one_over_two_pi = 1.0f / two_pi;

	float* Ip = new float[m_cameraSize];
	float* Idp = new float[m_cameraSize];
	memset(Ip, 0, sizeof(Ip[0]) * m_cameraSize);
	memset(Idp, 0, sizeof(Idp[0]) * m_cameraSize);

	float* cosN = new float[nStep];
	float* sinN = new float[nStep];
	for (int N = 0; N < nStep; N++)
	{
		cosN[N] = cos(two_pi * N / nStep);
		sinN[N] = sin(two_pi * N / nStep);
	}
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		float c = 0;
		float s = 0;
		float Isum = 0;
		for (int N = 0; N < nStep; N++)
		{
			c += fringeImage[N][idx] * cosN[N];
			s += fringeImage[N][idx] * sinN[N];
			Isum += float(fringeImage[N][idx]);			
		}
		wrapphase[idx] = (float)(-atan2(s, c)) * one_over_two_pi + 0.5f;
		Ip[idx] = Isum / nStep;
		Idp[idx] = sqrt(s * s + c * c) * 2.0f / nStep;
	}
	if (Iavg)
	{
		memcpy(Iavg, Ip, sizeof(Iavg[0]) * m_cameraSize);
		
	}
	if (amplitude)
	{
		memcpy(amplitude, Idp, sizeof(amplitude[0]) * m_cameraSize);
	}
	delete[] Ip;
	delete[] Idp;
	delete[] cosN;
	delete[] sinN;
}

void CPhaseWrapUnwrap::unwrapThreeFreqPhaseLine(float* unwrappedPhase, unsigned char* fringeImage[],
	int fringePeriods[], int fringeSteps[], int spikeFilterSize, float* Iavg, float* amplitude)
{
	double pi = atan2(1.0, 0.0) * 2.0;
	int T1 = fringePeriods[0];
	int T2 = fringePeriods[1];
	int T3 = fringePeriods[2];
	int nStep1 = fringeSteps[0];
	int nStep2 = fringeSteps[1];
	int nStep3 = fringeSteps[2];

	float* ph1 = new float[m_cameraSize];
	float* ph2 = new float[m_cameraSize];
	float* ph3 = new float[m_cameraSize];
	float* uph2 = new float[m_cameraSize];
	float* uphf = new float[m_cameraSize];

	// wrap high-freq phase
	computePhase(ph1, fringeImage, nStep1, Iavg, amplitude);

	// wrap middle-freq phase
	computePhase(ph2, &fringeImage[nStep1], nStep2);

	// wrap low-freq phase
	computePhase(ph3, &fringeImage[nStep1 + nStep2], nStep3);

	// unwrap middle-freq phase
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		float k2 = (ph3[idx]) * T3 / T2 - ph2[idx];
		uph2[idx] = ph2[idx] + round(k2);
	}

	// unwrap high-freq phase
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		// unwrap middle-freq phase
		float k1 = (uph2[idx]) * T2 / T1 - ph1[idx];
		unwrappedPhase[idx] = ph1[idx] + round(k1);
	}

	CImageFilters filter(m_cameraWidth, m_cameraHeight);
	// remove spikes
	filter.medianPhaseFilter(unwrappedPhase, uphf, spikeFilterSize);
	float phaseToPixels = (float)(T1);
	// remove spikes and convert phase to pixels
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		int k = round((uphf[idx] - unwrappedPhase[idx]));
		unwrappedPhase[idx] += (float)(k);
		unwrappedPhase[idx] *= phaseToPixels;
	}

	delete[] uph2;
	delete[] uphf;
	delete[] ph1;
	delete[] ph2;
	delete[] ph3;
}

void CPhaseWrapUnwrap::unwrapThreeFreqPhaseLine(float* unwrappedPhase, unsigned char** fringeImage, 
	vector<int> fringePeriods, vector<int>fringeSteps, int spikeFilterSize, float* Iavg, float* amplitude)
{
	if (fringePeriods.size() < 3 || fringeSteps.size() < 3)
	{
		cout << "fringe periods or fringe steps does not have enough elements (3)" << endl;
		return;
	}
	
	float* ph1 = new float[m_cameraSize];
	float* ph2 = new float[m_cameraSize];
	float* ph3 = new float[m_cameraSize];
	float* uph2 = new float[m_cameraSize];
	float* uphf = new float[m_cameraSize];

	int T1 = fringePeriods[0];
	int T2 = fringePeriods[1];
	int T3 = fringePeriods[2];
	int nStep1 = fringeSteps[0];
	int nStep2 = fringeSteps[1];
	int nStep3 = fringeSteps[2];


	// wrap high-freq phase
	computePhase(ph1, fringeImage, nStep1, Iavg, amplitude);

	// wrap middle-freq phase
	computePhase(ph2, fringeImage + nStep1, nStep2);

	// wrap low-freq phase
	computePhase(ph3, fringeImage + nStep1 + nStep2, nStep3);

	// unwrap middle-freq phase
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		float k2 = (ph3[idx]) * T3 / T2 - ph2[idx];
		uph2[idx] = ph2[idx] + round(k2);
	}

	// unwrap high-freq phase
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		// unwrap middle-freq phase
		float k1 = (uph2[idx]) * T2 / T1 - ph1[idx];
		unwrappedPhase[idx] = ph1[idx] + round(k1);
	}
	
	CImageFilters filter(m_cameraWidth, m_cameraHeight);
	// remove spikes
	filter.medianPhaseFilter(unwrappedPhase, uphf, spikeFilterSize);
	float phaseToPixels = (float)(T1);
	// remove spikes and convert phase to pixels
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		int k = round((uphf[idx] - unwrappedPhase[idx]));
		unwrappedPhase[idx] += (float)(k);
		unwrappedPhase[idx] *= phaseToPixels;
	}

	delete[] uph2;
	delete[] uphf;
	delete[] ph1;
	delete[] ph2;
	delete[] ph3;
}

void CPhaseWrapUnwrap::getFringeContrast(float* Iavg, float* amplitude, float* fringeContrast)
{
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		if (Iavg[idx] == 0)
		{
			fringeContrast[idx] = 0;
		}
		else
		{
			fringeContrast[idx] = amplitude[idx] / Iavg[idx];
		}
		
	}
}

void CPhaseWrapUnwrap::addFringeContrast(float* fringeContrast_1, float* fringeContrast_2, float* fringeContrast_out)
{
	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		fringeContrast_out[idx] = (fringeContrast_1[idx] + fringeContrast_2[idx]) / 2;
	}

	/*float max_contrast = *max_element(fringeContrast_out, fringeContrast_out + m_cameraSize);
	float min_contrast = *min_element(fringeContrast_out, fringeContrast_out + m_cameraSize);
	float difference = max_contrast - min_contrast;

	for (int idx = 0; idx < m_cameraSize; idx++)
	{
		fringeContrast_out[idx] = (fringeContrast_out[idx] - min_contrast) / difference;
	}*/
}


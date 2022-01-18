#include "CFileIO.h"

/*
*
*/
void CFileIO::WritePhaseMapFile(const char * nFileName, float * nPhaseData, int nImageWidth, int nImageHeight)
{
	/*ofstream File;
	File.open(nFileName);

	for (int i = 0; i < nImageHeight; i++)
	{
		for (int j = 0; j < nImageWidth; j++)
		{
			File << nPhaseData[i*nImageWidth + j] << " ";
		}
		File << std::endl;
	}

	File.close();*/
	FILE* fp;
	fopen_s(&fp, nFileName, "wb");
	if (fp)
	{
		fwrite(&nImageWidth, sizeof(int), 1, fp);
		fwrite(&nImageHeight, sizeof(int), 1, fp);
		fwrite(nPhaseData, sizeof(nPhaseData[0]), nImageWidth * nImageHeight, fp);
		fclose(fp);
	}
	else
	{
		cout << "cannot open file : " << nFileName << endl;
	}

}

void CFileIO::WriteArray(const char* nFileName, const int * nArray, int size)
{
	ofstream File;
	File.open(nFileName);
	for (int i = 0; i < size; i++)
	{
		File << nArray[i] << endl;
	}

	File.close();
 }

/*
*
*/
void CFileIO::Write3DPoints(const char * nFileName, const cv::Mat & nMatrix)
{
	ofstream File;
	File.open(nFileName);

	for (int i = 0; i < nMatrix.cols; i++)
	{
		File << i + 1 << ": [    ";
		for (int j = 0; j < nMatrix.rows; j++)
		{
			File << nMatrix.at<double>(j, i) << "    ";
		}
		File << "]" << std::endl;
	}

	File.close();
}


/*
*
*/
void CFileIO::WritePointVector3D(const char * nFileName, const vector<cv::Point3f> & nPoints)
{
	ofstream File;
	File.open(nFileName);

	for (int i = 0; i < nPoints.size(); i++)
	{
		File << nPoints[i] << endl;
	}

	File.close();
}

/*
*
*/
void CFileIO::WritePointVector2D(const char * nFileName, const vector<cv::Point2f> & nPoints)
{
	ofstream File;
	File.open(nFileName);

	for (int i = 0; i < nPoints.size(); i++)
	{
		File << nPoints[i] << endl;
	}

	File.close();
}



/*
*
*/
void CFileIO::WriteMatrix(const char * nFileName, const cv::Mat & nMatrix)
{
	ofstream File;
	File.open(nFileName);
	File << setiosflags(ios::fixed) << setiosflags(ios::left) << setprecision(6);
	for (int i = 0; i < nMatrix.rows; i++)
	{
		for (int j = 0; j < nMatrix.cols; j++)
		{
			File << setw(15) << nMatrix.at<double>(i, j);
		}
		File << endl;
	}

	File.close();
}

/*
*
*/
void CFileIO::WriteImageFile(const char* nFileName, const cv::Mat& nImage)
{
	ofstream File;
	File.open(nFileName);
	for (int i = 0; i < nImage.rows; i++)
	{
		for (int j = 0; j < nImage.cols; j++)
		{
			File << int(nImage.at<unsigned char>(i, j));
			if (j != nImage.cols - 1)
			{
				File << " ";
			}
		}
		File << endl;
	}
}
#include "image_alignment.h"


/*
* @brief
* @param
*/
void warp_affine(Mat& image, Mat& warp, Mat* output)
{
	assert(image.data != nullptr);
	assert(warp.rows == 2 && warp.cols == 3);

	Mat_<double> warp_matrix;
	warp.convertTo(warp_matrix, CV_64FC1);
	warp_matrix.at<double>(0, 0)++;
	warp_matrix.at<double>(1, 1)++;
	warpAffine(image, (*output), warp_matrix, image.size(), INTER_LINEAR | WARP_INVERSE_MAP);
}

/*
* @brief
* @param
*/
void gradient(Mat& image, Mat* grad_x, Mat* grad_y)
{
	Mat_<double> dImg;
	image.convertTo(dImg, CV_64FC1);
	int rows = image.rows;
	int cols = image.cols;

	Mat_<double> xTopVec = dImg.col(1) - dImg.col(0);
	Mat_<double> xBotVec = dImg.col(cols - 1) - dImg.col(cols - 2);
	Mat_<double> xForwMat = dImg(cv::Range(0, rows), cv::Range(0, cols - 2));
	Mat_<double> xBackMat = dImg(cv::Range(0, rows), cv::Range(2, cols));
	Mat_<double> centGx = (xBackMat - xForwMat) / 2;
	Mat_<double> tmpGx = cv::Mat::zeros(rows, cols, CV_64FC1);

	for (int i = 1; i < cols - 1; i++)
	{
		centGx.col(i - 1).copyTo(tmpGx.col(i));
	}
	xTopVec.copyTo(tmpGx.col(0));
	xBotVec.copyTo(tmpGx.col(cols - 1));

	Mat_<double> yTopArr = dImg.row(1) - dImg.row(0);
	Mat_<double> yBotArr = dImg.row(rows - 1) - dImg.row(rows - 2);
	Mat_<double> yForwMat = dImg(cv::Range(0, rows - 2), cv::Range(0, cols));
	Mat_<double> yBackMat = dImg(cv::Range(2, rows), cv::Range(0, cols));
	Mat_<double> centGy = (yBackMat - yForwMat) / 2;
	Mat_<double> tmpGy = cv::Mat::zeros(rows, cols, CV_64FC1);
	for (int i = 1; i < rows - 1; i++) {
		centGy.row(i - 1).copyTo(tmpGy.row(i));
	}
	yTopArr.copyTo(tmpGy.row(0));
	yBotArr.copyTo(tmpGy.row(rows - 1));

	tmpGx.copyTo((*grad_x));
	tmpGy.copyTo((*grad_y));
}


/*
* @brief
* @param
* jacobian = [x, 0, y, 0, 1, 0
			  0, x, 0, y, 0, 1]
*/
void compute_jacobian_affine(int height, int width, Mat* jacobian)
{
	assert(width > 0 && height > 0);

	Mat tmp_jacobian(2 * height, 6 * width, CV_16UC1);

	// x
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = j;
		}
	}

	// 0
	for (int i = 0; i < height; i++)
	{
		for (int j = width; j < 2 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = 0;
		}
	}

	// y
	for (int i = 0; i < height; i++)
	{
		for (int j = 2 * width; j < 3 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = i;
		}
	}

	//0
	for (int i = 0; i < height; i++)
	{
		for (int j = 3 * width; j < 4 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = 0;
		}
	}

	//1
	for (int i = 0; i < height; i++)
	{
		for (int j = 4 * width; j < 5 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = 1;
		}
	}

	//0
	for (int i = 0; i < height; i++)
	{
		for (int j = 5 * width; j < 6 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = 0;
		}
	}

	//0
	for (int i = height; i < 2 * height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = 0;
		}
	}

	//x
	for (int i = height; i < 2 * height; i++)
	{
		for (int j = width; j < 2 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = j - width;
		}
	}

	//0
	for (int i = height; i < 2 * height; i++)
	{
		for (int j = 2 * width; j < 3 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = 0;
		}
	}

	//y
	for (int i = height; i < 2 * height; i++)
	{
		for (int j = 3 * width; j < 4 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = i - height;
		}
	}

	//0
	for (int i = height; i < 2 * height; i++)
	{
		for (int j = 4 * width; j < 5 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = 0;
		}
	}

	//1
	for (int i = height; i < 2 * height; i++)
	{
		for (int j = 5 * width; j < 6 * width; j++)
		{
			tmp_jacobian.at<ushort>(i, j) = 1;
		}
	}

	tmp_jacobian.copyTo((*jacobian));
}

/*
* @brief
* @param
*/
void compute_sd_images(Mat& grad_x, Mat& grad_y, Mat& jacobian, int Np, vector<Mat>* sd_images)
{
	assert(Np > 0);
	assert(grad_x.size() == grad_y.size());
	assert(jacobian.rows == 2 * grad_x.rows && jacobian.rows == 2 * grad_y.rows);
	assert(jacobian.cols == Np * grad_x.cols && jacobian.cols == Np * grad_y.cols);
	assert((grad_x.type() == CV_64FC1) && (grad_y.type() == CV_64FC1) && (jacobian.type() == CV_16UC1));

	int width = grad_x.cols;
	int height = grad_x.rows;

	//Mat Tx(height, width, CV_16UC1);
	//Mat Ty(height, width, CV_16UC1);
	double Tx;
	double Ty;

	for (int p = 0; p < Np; p++)
	{
		Mat sd_image(height, width, CV_64FC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Tx = grad_x.at<double>(i, j) * jacobian.at<ushort>(i, j + p * width);
				Ty = grad_y.at<double>(i, j) * jacobian.at<ushort>(i + height, j + p * width);
				sd_image.at<double>(i, j) = Tx + Ty;
			}
		}
		sd_images->push_back(sd_image);
	}
}


/*
* @brief: Compute steepest descent images with weight. Used in the ICIA algorithm with weighted L2 norm
* @param
*/
void weight_sd_images(vector<Mat>& sd_images, Mat& weight, vector<Mat>* weighted_sd_images)
{
	assert(sd_images.size() > 0);
	assert(sd_images[0].rows == weight.rows && sd_images[0].cols == weight.cols);
	assert(sd_images[0].type() == CV_64FC1 && weight.type() == CV_64FC1);

	int Np = sd_images.size();
	for (int n = 0; n < Np; n++)
	{
		Mat new_sd_images = sd_images[n].mul(weight);
		weighted_sd_images->push_back(new_sd_images.clone());
	}
}

/*
* @brief: hessian matrix approximation with Gauss-Newton method
* @param
*/
void compute_hessian(vector<Mat>& sd_images, int Np, Mat* hessian)
{
	assert(Np > 0);
	assert(sd_images[0].type() == CV_64FC1);

	Mat tmp_hessian(Np, Np, CV_64FC1);

	for (int p = 0; p < Np; p++)
	{
		for (int k = 0; k < Np; k++)
		{
			Mat h = sd_images[p].mul(sd_images[k]);
			Scalar sum_val = sum(h);
			tmp_hessian.at<double>(k, p) = static_cast<double>(sum_val[0]);
		}
	}

	tmp_hessian.copyTo((*hessian));
}


/*
* @brief
* @param
*/
void compute_weighted_hessian(vector<Mat>& sd_images, Mat& weight, int Np, Mat* hessian)
{
	assert(Np > 0);
	assert(sd_images[0].type() == CV_64FC1 && weight.type() == CV_64FC1);

	Mat tmp_hessian(Np, Np, CV_64FC1);

	for (int p = 0; p < Np; p++)
	{
		for (int k = 0; k < Np; k++)
		{
			Mat h = sd_images[p].mul(sd_images[k]);
			h = h.mul(weight);
			Scalar sum_val = sum(h);
			tmp_hessian.at<double>(k, p) = static_cast<double>(sum_val[0]);
		}
	}

	tmp_hessian.copyTo((*hessian));
}


/*
* @brief
* @param
*/
void sd_update(vector<Mat>& sd_images, Mat& error_image, int Np, Mat* sd_delta_p)
{
	assert(Np > 0);
	assert(sd_images.size() == Np);
	assert(error_image.type() == CV_64FC1 && sd_images[0].type() == CV_64FC1);

	Mat tmp_sd_delta_p(Np, 1, CV_64FC1);

	for (int p = 0; p < Np; p++)
	{
		Mat tmp = sd_images[p].mul(error_image);
		Scalar sum_val = sum(tmp);
		tmp_sd_delta_p.at<double>(p, 0) = static_cast<double>(sum_val[0]);
	}

	tmp_sd_delta_p.copyTo((*sd_delta_p));
}

/*
* @brief
* @param
*/
void affine_param_update(Mat& warp_p, Mat& delta_p)
{
	assert(warp_p.rows == 2 && warp_p.cols == 3);
	assert(delta_p.rows == 6 && delta_p.cols == 1);
	assert(warp_p.type() == CV_64FC1 && delta_p.type() == CV_64FC1);

	// extend affine warp to 3x3 matrix
	Mat delta_p_extend = Mat::eye(3, 3, CV_64FC1);
	Mat warp_p_extend = Mat::eye(3, 3, CV_64FC1);

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			delta_p_extend.at<double>(i, j) = delta_p.at<double>(i + 2 * j, 0);
			warp_p_extend.at<double>(i, j) = warp_p.at<double>(i, j);
		}
	}

	delta_p_extend.at<double>(0, 0)++;
	delta_p_extend.at<double>(1, 1)++;
	warp_p_extend.at<double>(0, 0)++;
	warp_p_extend.at<double>(1, 1)++;

	// inverse compositional warp
	Mat delta_p_extend_inv = delta_p_extend.inv();

	// update
	Mat update_matrix = warp_p_extend * delta_p_extend_inv;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			warp_p.at <double>(i, j) = update_matrix.at<double>(i, j);
		}
	}
	warp_p.at<double>(0, 0)--;
	warp_p.at<double>(1, 1)--;
}


/*
* @brief: basic ICIA(inverse compositional image alignment) algorithm implementation
* @param: source: the image waiting to be aligned
*         target: the reference
*         warp_param: computed parameters (output)
*/
void inverse_compositional_align(Mat& source, Mat& target, Mat* warp_param, int max_iter, double criterion)
{
	double prev_rmse;
	double curr_rmse;
	int Np = 6;
	int height = source.rows;
	int width = source.cols;
	Mat warp = Mat::zeros(2, 3, CV_64FC1);
	warp.at<double>(0, 2) += 0.5;
	warp.at<double>(1, 2) += 0.5;
	Mat_<double> prev_warp;

	// pre-computation stage
	// (1) pre-compute image gradient of the reference
	Mat grad_x;
	Mat grad_y;
	//Sobel(target, grad_x, CV_64FC1, 1, 0);
	//Sobel(target, grad_y, CV_64FC1, 0, 1);
	// use self implemented gradient
	gradient(target, &grad_x, &grad_y);

	// (2) pre-compute jacobian
	Mat jacobian;
	compute_jacobian_affine(height, width, &jacobian);

	// (3) pre-compute steepest descent images of the reference
	vector<Mat> sd_images;
	compute_sd_images(grad_x, grad_y, jacobian, Np, &sd_images);

	// (4) pre-compute hessian matrix
	Mat hessian;
	compute_hessian(sd_images, Np, &hessian);
	Mat hessian_inv = hessian.inv();

	cout << "start iteration" << endl;
	// iteration stage
	for (int iter = 0; iter < max_iter; iter++)
	{
		Mat warp_image;
		// (1) compute warped image with current warp matrix
		warp_affine(source, warp, &warp_image);

		// (2) compute error image and RMSE
		Mat warp_image_double;
		Mat target_double;
		warp_image.convertTo(warp_image_double, CV_64FC1);
		target.convertTo(target_double, CV_64FC1);
		Mat error_image = warp_image_double - target_double;
		curr_rmse = norm(error_image) / sqrt(width * height);
		cout << curr_rmse << endl;
		/*if (curr_rmse <= criterion)
		{
			break;
		}*/
		if (iter > 0 && (prev_rmse - curr_rmse < criterion))
		{
			if (prev_rmse < curr_rmse)
			{
				warp = prev_warp;
				if (iter == 1)
				{
					warp.at<double>(0, 2) -= 0.5;
					warp.at<double>(1, 2) -= 0.5;
				}
			}
			break;
		}
		prev_rmse = curr_rmse;
		warp.copyTo(prev_warp);

		// (3) compute steepest descent parameter updates
		Mat sd_delta_p;
		sd_update(sd_images, error_image, Np, &sd_delta_p);

		// (4) compute parameter updates
		Mat delta_p = hessian_inv * sd_delta_p;

		// (5) matrix parameters update
		affine_param_update(warp, delta_p);
	}
	cout << "finish" << endl;
	warp.copyTo(*warp_param);
}


/*
* @brief: ICIA(inverse compositional image alignment) algorithm with weighted L2 norm implementation
* @param: source: the image waiting to be aligned
*         target: the reference
*         weight: weight paramters
*         warp_param: computed parameters (output)
*/
void weighted_inverse_compositional_align(Mat& source, Mat& target, Mat& weight, Mat* warp_param, int max_iter, double criterion)
{
	double prev_rmse = 100000;
	double curr_rmse;
	int Np = 6;
	int height = source.rows;
	int width = source.cols;
	Mat warp = Mat::zeros(2, 3, CV_64FC1);
	warp.at<double>(0, 2) += 0.5;
	warp.at<double>(1, 2) += 0.5;
	Mat_<double> prev_warp;

	if (weight.type() != CV_64FC1)
	{
		weight.convertTo(weight, CV_64FC1);
	}

	// pre-computation stage
	// (1) pre-compute image gradient of the reference
	Mat grad_x;
	Mat grad_y;
	//Sobel(target, grad_x, CV_64FC1, 1, 0);
	//Sobel(target, grad_y, CV_64FC1, 0, 1);
	// use self implemented gradient
	gradient(target, &grad_x, &grad_y);

	// (2) pre-compute jacobian
	Mat jacobian;
	compute_jacobian_affine(height, width, &jacobian);

	// (3) pre-compute weighted steepest descent images of the reference
	vector<Mat> sd_images;
	compute_sd_images(grad_x, grad_y, jacobian, Np, &sd_images);

	// (4) add weight to sd images
	vector<Mat> weighted_sd_images;
	weight_sd_images(sd_images, weight, &weighted_sd_images);

	// (5) pre-compute weighted hessian matrix
	Mat hessian;
	compute_weighted_hessian(sd_images, weight, Np, &hessian);
	Mat hessian_inv = hessian.inv();

	cout << "start iteration" << endl;
	// iteration stage
	for (int iter = 0; iter < max_iter; iter++)
	{
		Mat warp_image;
		// (1) compute warped image with current warp matrix
		warp_affine(source, warp, &warp_image);

		// (2) compute error image and RMSE
		Mat warp_image_double;
		Mat target_double;
		warp_image.convertTo(warp_image_double, CV_64FC1);
		target.convertTo(target_double, CV_64FC1);
		Mat error_image = warp_image_double - target_double;
		curr_rmse = norm(error_image) / sqrt(width * height);
		cout << prev_rmse << curr_rmse << endl;
		/*if (curr_rmse <= criterion)
		{
			break;
		}*/
		if (iter > 0 && (prev_rmse - curr_rmse < criterion))
		{
			if (prev_rmse < curr_rmse)
			{
				warp = prev_warp;
				if (iter == 1)
				{
					warp.at<double>(0, 2) -= 0.5;
					warp.at<double>(1, 2) -= 0.5;
				}
			}
			break;
		}
		prev_rmse = curr_rmse;
		warp.copyTo(prev_warp);
		// (3) compute steepest descent parameter updates
		Mat sd_delta_p;
		sd_update(weighted_sd_images, error_image, Np, &sd_delta_p);

		// (4) compute parameter updates
		Mat delta_p = hessian_inv * sd_delta_p;

		// (5) matrix parameters update
		affine_param_update(warp, delta_p);
	}

	warp.copyTo(*warp_param);

}


/*
* @brief: test the ICIA algorithm.
*         warp the image then use ICIA to warp the image back, compare the parameters
* @param: test_image
*         warp_test: use to warp test_image
*/
void test(Mat& test_image, Mat& warp_test)
{
	// warp test image
	Mat warp_image;
	warp_affine(test_image, warp_test, &warp_image);
	//imshow("warp_image", warp_image);
	//waitKey();
	//destroyWindow("warp_image");
	imwrite("../../data/single_image_test/warp_test_image.png", warp_image);

	// warp back
	Mat warp_icia;
	inverse_compositional_align(test_image, warp_image, &warp_icia, 100, 0);

	cout << "comparing params" << endl;
	cout << "input warp params:" << endl;
	cout << warp_test << endl;
	cout << "ICIA output:" << endl;
	cout << warp_icia << endl;

	//Mat icia_estimate;
	//warpAffine(test_image, icia_estimate, warp_icia, warp_image.size());
	//imwrite("../../data/test/warp_icia.png", icia_estimate);
}


/*
* @brief
* @param
*/
void align_image_stack(vector<Mat>& stack, vector<Mat>* warp_matrix)
{
	assert(stack.size() >= 2);
	
	int stackSize = stack.size();
	Mat warpToFirst = Mat::eye(3, 3, CV_64FC1);

	for (int stackIdx = 0; stackIdx < stackSize-1; stackIdx++)
	{
		Mat warp;
		inverse_compositional_align(stack[stackIdx+1], stack[stackIdx], &warp);
	
		warp.push_back(Mat::zeros(1, 3, CV_64FC1));
		warp += Mat::eye(3, 3, CV_64FC1);
		warpToFirst = warp * warpToFirst;
		Mat warpToFirstParam;
		warpToFirst.copyTo(warpToFirstParam);
		warpToFirstParam.pop_back();
		warpToFirstParam.at<double>(0, 0)--;
		warpToFirstParam.at<double>(1, 1)--;
		warp_matrix->push_back(warpToFirstParam);
	}
}
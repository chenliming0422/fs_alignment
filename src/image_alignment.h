#pragma once
/***********************************************************************************************
* Copyright (C), 2021.Purdue University Mechanical Engineering, XTZT Lab, All rights reserved.
* @file: image_alignment.h
* @brief: Header file for image_alignment.cpp
* @author: Liming Chen
* @email: chen3496@purdue.edu
* @version: V1.0
* @date: 2022-01-03
* @note
* @warning
* @todo
* @history:
***********************************************************************************************/
#include <vector>
#include <assert.h>
#include <iostream>
#include <math.h>

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

void warp_affine(Mat& image, Mat& warp, Mat* output);

void gradient(Mat& image, Mat* grad_x, Mat* grad_y);

void compute_jacobian_affine(int height, int width, Mat* jacobian);

void compute_sd_images(Mat& grad_x, Mat& grad_y, Mat& jacobian, int Np, vector<Mat>* sd_images);

void compute_weighted_sd_images(Mat& grad_x, Mat& grad_y, Mat& jacobian, Mat& weight, int Np, vector<Mat>* sd_images);

void compute_hessian(vector<Mat>& sd_images, int Np, Mat* hessian);

void compute_weighted_hessian(vector<Mat>& sd_images, Mat& weight, int Np, Mat* hessian);

void sd_update(vector<Mat>& sd_images, Mat& error_image, int Np, Mat* sd_delta_p);

void affine_param_update(Mat& warp_p, Mat& delta_p);

void inverse_compositional_align(Mat& source, Mat& target, Mat* warp_param, int max_iter = 20, double criterion = 0.0);

void weighted_inverse_compositional_align(Mat& source, Mat& target, Mat& weight, Mat* warp_param, int max_iter = 20, double criterion = 0.0);

void test(Mat& test_image, Mat& warp_test);


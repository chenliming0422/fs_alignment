# fs_alignment

This code implements image alignment for focal stack applications using the inverse compositional image alignment (ICIA) algorithm.

# environment

**1. Visual Studio 2019** 

**2. OpenCV 4.2.0**

PS: you may need to change the opencv path.

# test

The code in the folder **src** includes two tests:

## single image test

Copy following code in the main function:

```
	
	string imagePath = "../../data/signle_image_test/screw.png";
	Mat image = imread(imagePath, IMREAD_GRAYSCALE);
	imshow("image", image);
	waitKey();
	destroyWindow("image");

	Mat test_warp = 0.01 * Mat::eye(2, 3, CV_64FC1);
	test_warp.at<double>(0, 2) = 2;
	test_warp.at<double>(1, 2) = 3;

	test(image, test_warp);

	return 0;


```

You can also change the numbers in the matrix **test_warp** in the code.

## image stack test

Run the main function
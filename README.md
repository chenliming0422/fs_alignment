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
string imagePath = "../../data/single_image_test/screw.png";
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

Align all images to the first image in the image sequence. The main function in **main.cpp** now is for this application.

The image stack in the folder **Screw** is captured with different focal length, so different parts in images are in focus. The goal of this test is to align the stack in order to stitch them to form an all-in-focus image.
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

const int HIST_IMG_HEIGHT = 400;

void createHistogram(Mat& img, Mat& hist)
{
    long counts[256] = {}; // Zero-initialized array

    // Count pixel intensities
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            int val = img.at<uchar>(y, x);
            counts[val]++;
        }
    }

    // Find the max count
    long max_count = 0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] > max_count) max_count = counts[i];
    }

    // Create histogram image
    hist = Mat(HIST_IMG_HEIGHT, 512, CV_8UC1, Scalar(255)); // white background

    for (int i = 0; i < 256; i++) {
        int height = static_cast<int>((double)counts[i] / max_count * HIST_IMG_HEIGHT);
        if (height == 0) continue;

        int x1 = i * 2;
        int x2 = x1 + 1;
        int y1 = HIST_IMG_HEIGHT - 1;
        int y2 = HIST_IMG_HEIGHT - height;

        // Draw 2-pixel wide black bar
        line(hist, Point(x1, y1), Point(x1, y2), Scalar(0), 1);
        line(hist, Point(x2, y1), Point(x2, y2), Scalar(0), 1);
    }
}




int main(int argc, char *argv[])
{
	Mat img;
	Mat hist;

	img = imread(argv[1], IMREAD_GRAYSCALE);

	if (img.empty()) {
		printf("Failed to load image '%s'\n", argv[1]);
		return -1;
	} else {
		printf("Loaded image '%s' successfully! Size: %dx%d\n", 
			argv[1], img.cols, img.rows);
	}


	// Check if the image was successfully loaded
	if (img.empty()) {
			printf("Failed to load image '%s'\n", argv[1]);
			return -1;
	}

	// Create image histogram
	createHistogram(img, hist);

	namedWindow("Histogram", WINDOW_NORMAL);
	imwrite("hist.jpg", hist);

	imshow("Histogram", hist);

	// Wait for a key press before quitting
	waitKey(0);

	return 0;
}

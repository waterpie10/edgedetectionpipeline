#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Create a 400 x 512 histogram image from a grayscale input (Task 3.1 logic)
void createHistogram(const cv::Mat& img, cv::Mat& hist)
{
    const int HIST_IMG_HEIGHT = 400;
    // 1) Count occurrences of each intensity [0..255]
    long counts[256] = {0};
    for(int r = 0; r < img.rows; r++)
    {
        const uchar* rowPtr = img.ptr<uchar>(r);
        for(int c = 0; c < img.cols; c++)
        {
            int val = rowPtr[c];
            counts[val]++;
        }
    }

    // 2) Find maximum count
    long maxCount = 0;
    for(int i = 0; i < 256; i++)
    {
        if(counts[i] > maxCount) {
            maxCount = counts[i];
        }
    }

    // 3) Create a 400 x 512 single-channel image, fill with white
    hist = cv::Mat(HIST_IMG_HEIGHT, 512, CV_8UC1, cv::Scalar(255));

    // 4) Scale so the largest bar is at the top
    double scale = static_cast<double>(HIST_IMG_HEIGHT) / (double)maxCount;

    // 5) Draw each bar (2 pixels wide)
    for(int i = 0; i < 256; i++)
    {
        int barHeight = static_cast<int>(counts[i] * scale);
        if (barHeight > 0)
        {
            int x1 = i * 2;
            int x2 = x1 + 1;
            int y1 = HIST_IMG_HEIGHT - barHeight;
            int y2 = HIST_IMG_HEIGHT - 1;
            // Black bar on white background
            cv::rectangle(hist,
                          cv::Point(x1, y1),
                          cv::Point(x2, y2),
                          cv::Scalar(0),
                          cv::FILLED);
        }
    }
}

int main(int argc, char** argv)
{
    // Usage: ./threshold_single <image> [<threshold>]
    // Example:
    //    ./threshold_single fundus.tif 80

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image> [<threshold>]\n";
        return -1;
    }

    std::string filename = argv[1];
    // Default to 128 if no threshold specified
    int thresholdVal = 128;
    if (argc >= 3) {
        thresholdVal = std::stoi(argv[2]);
    }

    // 1) Load image in grayscale
    cv::Mat imgGray = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if(imgGray.empty()) {
        std::cerr << "Error: could not load " << filename << std::endl;
        return -1;
    }
    std::cout << "Loaded: " << filename
              << " (size: " << imgGray.cols << "x" << imgGray.rows << ")\n";

    // 2) Create & save histogram
    cv::Mat histImg;
    createHistogram(imgGray, histImg);
    std::string histOut = filename + "_hist.jpg";
    cv::imwrite(histOut, histImg);
    std::cout << "Saved histogram to " << histOut << std::endl;

    // 3) Apply threshold
    cv::Mat threshImg;
    cv::threshold(imgGray, threshImg, thresholdVal, 255, cv::THRESH_BINARY);

    // 4) Save thresholded result
    std::string binOut = filename + "_thresholded.jpg";
    cv::imwrite(binOut, threshImg);
    std::cout << "Saved thresholded image to " << binOut
              << " (threshold=" << thresholdVal << ")\n";

    // 5) Display them so you can examine results
    cv::imshow("Histogram (" + filename + ")", histImg);
    cv::imshow("Thresholded (" + filename + ")", threshImg);

    // 6) Wait for a key press
    std::cout << "Press any key in the image window to exit...\n";
    cv::waitKey(0);

    return 0;
}

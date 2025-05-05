#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

// Trackbar-controlled parameters
int g_blurKsize     = 7;   
int g_cannyLower    = 50;
int g_cannyUpper    = 150;
int g_houghThresh   = 50;
int g_minLineLen    = 30;
int g_maxLineGap    = 20;
int g_verticalDelta = 10;  

cv::Mat g_colorImg;  // Original color image

// Global Mats for each intermediate step
cv::Mat g_edges;
cv::Mat g_allLines;
cv::Mat g_shortLinesRemoved;    
cv::Mat g_onlyHorizontalLines;  
cv::Mat g_horizonDraw;

// ----------------------------------------------
// 1) fitPoly: polynomial regression (no VLA)
// ----------------------------------------------
std::vector<double> fitPoly(const std::vector<cv::Point>& points, int n)
{
    int nPoints = (int) points.size();
    std::vector<double> xVals(nPoints), yVals(nPoints);

    for(int i = 0; i < nPoints; i++)
    {
        xVals[i] = points[i].x;
        yVals[i] = points[i].y;
    }

    // Build an (n+1) x (n+2) augmented matrix
    std::vector<std::vector<double>> mat(n+1, std::vector<double>(n+2, 0.0));
    for(int row = 0; row < n+1; row++)
    {
        for(int col = 0; col < n+1; col++)
        {
            double sumVal = 0.0;
            for(int i = 0; i < nPoints; i++)
                sumVal += std::pow(xVals[i], row + col);
            mat[row][col] = sumVal;
        }

        double sumVal2 = 0.0;
        for(int i = 0; i < nPoints; i++)
            sumVal2 += std::pow(xVals[i], row) * yVals[i];
        mat[row][n+1] = sumVal2;
    }

    // Solve via Gauss elimination
    std::vector<double> coeffVec(n+1, 0.0);
    for(int i = 0; i < n; i++)
    {
        for(int k = i+1; k <= n; k++)
        {
            double t = mat[k][i] / mat[i][i];
            for(int j = 0; j <= n+1; j++)
                mat[k][j] -= t * mat[i][j];
        }
    }
    for(int i = n; i >= 0; i--)
    {
        coeffVec[i] = mat[i][n+1];
        for(int j = 0; j < n+1; j++)
        {
            if(j != i)
                coeffVec[i] -= mat[i][j] * coeffVec[j];
        }
        coeffVec[i] /= mat[i][i];
    }
    return coeffVec;
}

// ----------------------------------------------
// 2) pointAtX: evaluate polynomial
// ----------------------------------------------
cv::Point pointAtX(const std::vector<double>& coeff, double x)
{
    double y = 0.0;
    for(int i = 0; i < (int)coeff.size(); i++)
        y += std::pow(x, i) * coeff[i];
    return cv::Point((int)std::round(x), (int)std::round(y));
}

// ----------------------------------------------
// 3) The pipeline that runs each time a trackbar changes
// ----------------------------------------------
void runHorizonDetection()
{
    if(g_colorImg.empty())
    {
        std::cerr << "No image loaded!\n";
        return;
    }

    // 1) Convert to grayscale + blur
    cv::Mat gray;
    cv::cvtColor(g_colorImg, gray, cv::COLOR_BGR2GRAY);

    // Enforce odd kernel size
    if(g_blurKsize < 1) g_blurKsize = 1;
    if(g_blurKsize % 2 == 0) g_blurKsize += 1;
    cv::GaussianBlur(gray, gray, cv::Size(g_blurKsize, g_blurKsize), 0);

    // 2) Canny
    cv::Canny(gray, g_edges, g_cannyLower, g_cannyUpper);
    cv::imshow("Canny Edges", g_edges);

    // 3) Hough
    std::vector<cv::Vec4i> linesP;
    double rho   = 1.0;
    double theta = CV_PI / 180.0;
    cv::HoughLinesP(g_edges, linesP, rho, theta,
                    g_houghThresh, (double)g_minLineLen, (double)g_maxLineGap);

    // 3a) Draw all lines
    g_allLines = g_colorImg.clone();
    for(const auto& ln : linesP)
    {
        cv::line(g_allLines, cv::Point(ln[0], ln[1]),
                             cv::Point(ln[2], ln[3]),
                             cv::Scalar(0,0,255), 1);
    }
    cv::imshow("All Hough Lines", g_allLines);

    // 4) STEP ONE: Remove short lines only
    std::vector<cv::Vec4i> linesAfterShortRemoval;
    g_shortLinesRemoved = g_colorImg.clone();
    for(const auto& ln : linesP)
    {
        int x1 = ln[0], y1 = ln[1];
        int x2 = ln[2], y2 = ln[3];
        double dx = (double)(x2 - x1);
        double dy = (double)(y2 - y1);
        double length = std::sqrt(dx*dx + dy*dy);

        if(length >= (double)g_minLineLen)
        {
            // keep it
            linesAfterShortRemoval.push_back(ln);
            cv::line(g_shortLinesRemoved, cv::Point(x1, y1), cv::Point(x2, y2),
                     cv::Scalar(255, 0, 0), 2); // blue
        }
    }
    cv::imshow("Short Lines Removed", g_shortLinesRemoved);

    // 5) STEP TWO: Remove near-vertical from the short-removed set
    std::vector<cv::Vec4i> finalLines;
    g_onlyHorizontalLines = g_colorImg.clone();

    for(const auto& ln : linesAfterShortRemoval)
    {
        int x1 = ln[0], y1 = ln[1];
        int x2 = ln[2], y2 = ln[3];

        if(std::abs(x2 - x1) >= g_verticalDelta)
        {
            // keep
            finalLines.push_back(ln);
            cv::line(g_onlyHorizontalLines, cv::Point(x1, y1), cv::Point(x2, y2),
                     cv::Scalar(255, 0, 0), 2); // still blue
        }
    }
    cv::imshow("Only Horizontal Lines", g_onlyHorizontalLines);

    // 6) Fit a polynomial with final lines
    //    Collect endpoints
    std::vector<cv::Point> horizonPoints;
    for(const auto& ln : finalLines)
    {
        horizonPoints.push_back(cv::Point(ln[0], ln[1]));
        horizonPoints.push_back(cv::Point(ln[2], ln[3]));
    }

    if(horizonPoints.size() < 4)
    {
        // Not enough data
        g_horizonDraw = g_colorImg.clone();
        cv::putText(g_horizonDraw, "Not enough points!",
                    cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0,0,255), 2);
        cv::imshow("Fitted Horizon", g_horizonDraw);
        return;
    }

    std::vector<double> coeffs = fitPoly(horizonPoints, 2);

    // 7) Draw the polynomial
    g_horizonDraw = g_colorImg.clone();
    for(int x = 0; x < g_horizonDraw.cols; x++)
    {
        cv::Point pt = pointAtX(coeffs, (double)x);
        if(pt.y >= 0 && pt.y < g_horizonDraw.rows)
            cv::circle(g_horizonDraw, pt, 1, cv::Scalar(0,255,0), -1);
    }
    cv::imshow("Fitted Horizon", g_horizonDraw);
}

// ----------------------------------------------
// 4) onTrackbarChange
// ----------------------------------------------
static void onTrackbarChange(int, void*)
{
    runHorizonDetection();
}

// ----------------------------------------------
// 5) MAIN
// ----------------------------------------------
int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <image>\n";
        return -1;
    }

    // Load color image
    g_colorImg = cv::imread(argv[1], cv::IMREAD_COLOR);
    if(g_colorImg.empty())
    {
        std::cerr << "Could not load " << argv[1] << "\n";
        return -1;
    }

    // Create windows
    cv::namedWindow("Canny Edges",         cv::WINDOW_NORMAL);
    cv::namedWindow("All Hough Lines",     cv::WINDOW_NORMAL);
    cv::namedWindow("Short Lines Removed", cv::WINDOW_NORMAL);
    cv::namedWindow("Only Horizontal Lines", cv::WINDOW_NORMAL);
    cv::namedWindow("Fitted Horizon",      cv::WINDOW_NORMAL);

    cv::namedWindow("Controls", cv::WINDOW_NORMAL);

    // Create trackbars
    cv::createTrackbar("Blur Ksize",       "Controls", &g_blurKsize,     31, onTrackbarChange);
    cv::createTrackbar("Canny Lower",      "Controls", &g_cannyLower,    500, onTrackbarChange);
    cv::createTrackbar("Canny Upper",      "Controls", &g_cannyUpper,    500, onTrackbarChange);
    cv::createTrackbar("Hough Thresh",     "Controls", &g_houghThresh,   200, onTrackbarChange);
    cv::createTrackbar("Min Line Len",     "Controls", &g_minLineLen,    300, onTrackbarChange);
    cv::createTrackbar("Max Line Gap",     "Controls", &g_maxLineGap,    100, onTrackbarChange);
    cv::createTrackbar("Vert Delta",       "Controls", &g_verticalDelta, 50,  onTrackbarChange);

    // Run once
    runHorizonDetection();

    std::cout << "Adjust trackbars to tune parameters.\n";
    std::cout << "Press [s] to save all images. Press [ESC] to quit.\n";

    while(true)
    {
        int key = cv::waitKey(50);
        if(key == 27) // ESC
            break;
        else if(key == 's')
        {
            // Save them all
            cv::imwrite("edges_snapshot.jpg",              g_edges);
            cv::imwrite("all_lines_snapshot.jpg",          g_allLines);
            cv::imwrite("short_lines_removed_snapshot.jpg", g_shortLinesRemoved);
            cv::imwrite("only_horizontal_lines_snapshot.jpg", g_onlyHorizontalLines);
            cv::imwrite("fitted_horizon_snapshot.jpg",      g_horizonDraw);

            std::cout << "Saved images with params:\n"
                      << "   BlurKsize=" << g_blurKsize
                      << ", CannyLower=" << g_cannyLower
                      << ", CannyUpper=" << g_cannyUpper
                      << ", HoughThresh=" << g_houghThresh
                      << ", MinLineLen=" << g_minLineLen
                      << ", MaxLineGap=" << g_maxLineGap
                      << ", VertDelta=" << g_verticalDelta
                      << "\n";
        }
    }

    return 0;
}

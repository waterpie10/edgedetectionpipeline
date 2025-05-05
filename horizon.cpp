#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>


//Polynomial regression function
std::vector<double> fitPoly(const std::vector<cv::Point>& points, int n)
{
    int nPoints = (int) points.size();

    // Separate x and y values
    std::vector<double> xValues(nPoints), yValues(nPoints);
    for(int i = 0; i < nPoints; i++)
    {
        xValues[i] = points[i].x;
        yValues[i] = points[i].y;
    }

    // We'll create an (n+1) x (n+2) "augmented matrix":
    // That is matrixSystem[row][col],  row in [0..n], col in [0..n+1]
    std::vector<std::vector<double>> matrixSystem(n+1, std::vector<double>(n+2, 0.0));

    // Fill the augmented matrix
    for(int row = 0; row < n+1; row++)
    {
        for(int col = 0; col < n+1; col++)
        {
            double sumVal = 0.0;
            for(int i = 0; i < nPoints; i++)
            {
                sumVal += std::pow(xValues[i], row + col);
            }
            matrixSystem[row][col] = sumVal;
        }

        double sumVal2 = 0.0;
        for(int i = 0; i < nPoints; i++)
        {
            sumVal2 += std::pow(xValues[i], row) * yValues[i];
        }
        matrixSystem[row][n+1] = sumVal2;
    }

    // We'll store polynomial coefficients in coeffVec
    std::vector<double> coeffVec(n+1, 0.0);

    // 1) Gauss elimination
    for(int i = 0; i < n; i++)
    {
        for(int k = i+1; k <= n; k++)
        {
            double t = matrixSystem[k][i] / matrixSystem[i][i];
            for(int j = 0; j <= n+1; j++)
            {
                matrixSystem[k][j] -= t * matrixSystem[i][j];
            }
        }
    }

    // 2) Back-substitution
    for(int i = n; i >= 0; i--)
    {
        coeffVec[i] = matrixSystem[i][n+1];
        for(int j = 0; j < n+1; j++)
        {
            if(j != i)
            {
                coeffVec[i] -= matrixSystem[i][j] * coeffVec[j];
            }
        }
        coeffVec[i] /= matrixSystem[i][i];
    }

    return coeffVec;
}


//Returns the point for the equation determined
//by a vector of coefficents, at a certain x location
cv::Point pointAtX(std::vector<double> coeff, double x)
{
  double y = 0;
  for(int i = 0; i < coeff.size(); i++)
  y += pow(x, i) * coeff[i];
  return cv::Point(x, y);
}

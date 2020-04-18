
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, const std::vector<LidarPoint> &outliers, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot outliers
    for (auto it = outliers.begin(); it != outliers.end(); ++it)
    {
        // world coordinates
        float xw = (*it).x; // world position in m with x facing forward from sensor
        float yw = (*it).y; // world position in m with y facing left from sensor

        // top-view coordinates
        int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

        // draw individual point
        cv::circle(topviewImg, cv::Point(x, y), 5, {0, 0, 255}, -1);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}

static void lidarXMeanStddev(std::vector<LidarPoint> &points, double &mean, double &stddev)
{
    mean = 0.0;
    stddev = 0.0;

    // no points
    if (points.empty())
        return;

    const double n = points.size();

    // mean
    for (const auto &pnt : points)
    {
        mean += pnt.x / n;
    }

    // standard deviation
    if (n > 1)
    {
        double sse = 0.0;
        for (const auto &pnt : points)
        {
            const double err = (pnt.x - mean);
            sse += err * err;
        }
        stddev = std::sqrt(sse / (n - 1));
    }
}

std::vector<LidarPoint> computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // current outliers
    std::vector<LidarPoint> outliers;

    // filter outliers (2.0 corresponds to approx 5%, 1.645 to 10%)
    const double stddevFromMean = 1.7;

    // time between two measurements in seconds
    double dT = 1.0 / frameRate;

    // get mean and stddev on x position
    double meanXPrev, stddevXPrev, meanXCurr, stddevXCurr;
    lidarXMeanStddev(lidarPointsPrev, meanXPrev, stddevXPrev);
    lidarXMeanStddev(lidarPointsCurr, meanXCurr, stddevXCurr);

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (it->x > (meanXPrev - stddevFromMean * stddevXPrev)) {
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (it->x > (meanXCurr - stddevFromMean * stddevXCurr)) {
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        } else {
            outliers.push_back((*it));
        }
    }

    // compute TTC from both measurements
    if (minXPrev > (minXCurr + 1e-9)) {
        TTC = minXCurr * dT / (minXPrev - minXCurr);
    } else {
        // car is standing still or moving away
        TTC = -1.0;
    }

    return outliers;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // count number of matches inside bounding boxes
    // Data structure:
    // countMap = {
    //     [prevIndex]: { [currIndex]: matches }
    // }
    std::map<int, std::map<int, int>> countMap;

    // loop over all matches and associate them to a 2D bounding box
    for (auto match = matches.begin(); match != matches.end(); ++match)
    {

        // assemble vector for matrix-vector-multiplication
        cv::KeyPoint &prevKeypoint = prevFrame.keypoints[match->queryIdx];
        cv::KeyPoint &currKeypoint = currFrame.keypoints[match->trainIdx];

        // go through each bounding box in previous image
        for (auto prevBB = prevFrame.boundingBoxes.begin(); prevBB != prevFrame.boundingBoxes.end(); ++prevBB)
        {
            // check wether point is within previous bounding box
            if (prevBB->roi.contains(prevKeypoint.pt))
            {
                // go through each bounding box in current image
                for (auto currBB = currFrame.boundingBoxes.begin(); currBB != currFrame.boundingBoxes.end(); ++currBB)
                {

                    // check wether point is within current bounding box
                    if (currBB->roi.contains(currKeypoint.pt))
                    {
                        // add box IDs to count map
                        const int prevID = prevBB->boxID;
                        const int currID = currBB->boxID;

                        // new map item for previous index
                        if (!countMap.count(prevID))
                        {
                            countMap.insert({prevID, {}});
                        }

                        if (!countMap.at(prevID).count(currID))
                        {
                            // new map item for current index
                            countMap.at(prevID).insert({currID, 1});
                        }
                        else
                        {
                            // add to matches
                            countMap.at(prevID).at(currID) += 1;
                        }
                    }
                }
            }
        }
    }

    // find max count for each bounding box pairs
    for (const auto &prevPair : countMap)
    {
        const int prevID = prevPair.first;
        int bestCurrID = -1;
        int numMatches = 0;
        for (const auto &currPair : prevPair.second)
        {
            const int currID = currPair.first;
            const int count = currPair.second;
            if (count > numMatches)
            {
                bestCurrID = currID;
                numMatches = count;
            }
        }
        bbBestMatches.insert({prevID, bestCurrID});
    }
}

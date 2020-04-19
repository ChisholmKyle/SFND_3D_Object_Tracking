
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"

void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT, double &duration);
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches, double &duration);
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, double &duration);
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, const std::vector<LidarPoint> &outliers, const MatchingParameters &parameters, const std::string &imgPrefix);

std::vector<cv::KeyPoint> computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                                           std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, double &duration);
std::vector<LidarPoint> computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                                        std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, double &duration);
#endif /* camFusion_hpp */

#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType, double &duration)
{
    // configure matcher
    const double ssdRatioThreshold = 0.8;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        const bool crossCheck = (selectorType.compare("SEL_KNN") == 0 ? false : true);
        int normType = (descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2);
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F)
        {
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    double t = (double)cv::getTickCount();
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);
        for (int k = 0; k < knnMatches.size(); ++k)
        {
            // ssd ratio
            const float ratio = knnMatches[k][0].distance / knnMatches[k][1].distance;
            if (ratio < ssdRatioThreshold)
                matches.push_back(knnMatches[k][0]);
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    duration = 1000 * t / 1.0;

}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &duration)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {

        // int size = 32; // BRIEF descriptor size.

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        // int nfeatures = 500;        // The maximum number of features to retain.
        // float scaleFactor = 1.2f;   // Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid.
        // int nlevels = 8;            // The number of pyramid levels.
        // int edgeThreshold = 31;     // This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
        // cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE for Harris edge detection or less stable FAST_SCORE
        // int patchSize = 31;         // Size of the patch used by the oriented BRIEF descriptor.
        // int fastThreshold = 20;     // FAST threshold

        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        // bool  	orientationNormalized = true; // Enable orientation normalization
        // bool  	scaleNormalized = true;       // Enable scale normalization.
        // float  	patternScale = 22.0f;         // Scaling of the description pattern.
        // int  	nOctaves = 4;                 // Number of octaves covered by the detected keypoints.
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        // cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB; // Type of the extracted descriptor
        // int  	descriptor_size = 0;     // Size of the descriptor in bits. 0 -> Full size
        // int  	descriptor_channels = 3; // Number of channels in the descriptor (1, 2, 3)
        // float  	threshold = 0.001f;      // Detector response threshold to accept point
        // int  	nOctaves = 4;            // Maximum octave evolution of the image
        // int  	nOctaveLayers = 4;       // Default number of sublevels per scale level
        // cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type.

        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        // int nfeatures = 0;                // The number of best features to retain.
        // int nOctaveLayers = 3;            // The number of layers in each octave.
        // double contrastThreshold = 0.04;  // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        // double edgeThreshold = 10;        // The threshold used to filter out edge-like features
        // double sigma = 1.6;               // The sigma of the Gaussian applied to the input image at the octave #0

        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    duration = 1000 * t / 1.0;
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &duration, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    duration = 1000 * t / 1.0;

    // visualize results
    if (bVis)
    {
        cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << duration << " ms" << endl;
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &duration, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    int nmsSize = 5; // non-maximum suppression size

    // Apply corner detection
    double t = (double)cv::getTickCount();
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled, dst_thresh, dst_dilated, dst_maxima;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // dilate scaled image
    dst_dilated = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::Mat dilationKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                       cv::Size(2 * nmsSize + 1, 2 * nmsSize + 1),
                                                       cv::Point(nmsSize, nmsSize));
    cv::dilate(dst_norm_scaled, dst_dilated, dilationKernel);

    // threshold scaled image, anything below minResponse is 0
    dst_thresh = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::threshold(dst_norm_scaled, dst_thresh, minResponse, 255, cv::THRESH_TOZERO);

    // maxima: compare threshold to dilated
    dst_maxima = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::compare(dst_thresh, dst_dilated, dst_maxima, cv::CmpTypes::CMP_GE);

    // keypoints from contour analysis
    vector<vector<cv::Point>> contours;
    cv::findContours(dst_maxima, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // analyze contours
    for (const auto &contour : contours)
    {
        // centroid
        cv::Point2f centroid;
        cv::Moments moments = cv::moments(contour);
        if (moments.m00 == 0)
        {
            // mean of contour points
            for (const auto &point : contour)
            {
                centroid.x += point.x;
                centroid.y += point.y;
            }
            centroid.x /= contour.size();
            centroid.y /= contour.size();
        }
        else
        {
            centroid = cv::Point2f(
                moments.m10 / moments.m00,
                moments.m01 / moments.m00);
        }

        // value
        int response = dst_thresh.at<uchar>(centroid.x, centroid.y);

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = centroid;
        newKeyPoint.size = nmsSize + 1;
        newKeyPoint.response = response;

        keypoints.push_back(newKeyPoint);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    duration = 1000 * t / 1.0;
    // cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

//  FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, double &duration, bool bVis)
{

    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0)
    {
        // int threshold = 10;
        // bool nonmaxSuppression = true;
        // cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;

        detector = cv::FastFeatureDetector::create();
    }
    else if (detectorType.compare("BRISK") == 0)
    {

        // int threshold = 30;        // FAST/AGAST detection threshold score.
        // int octaves = 3;           // detection octaves (use 0 to do single scale)
        // float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        detector = cv::BRISK::create();
    }
    else if (detectorType.compare("ORB") == 0)
    {
        // int nfeatures = 500;        // The maximum number of features to retain.
        // float scaleFactor = 1.2f;   // Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid.
        // int nlevels = 8;            // The number of pyramid levels.
        // int edgeThreshold = 31;     // This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
        // cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE for Harris edge detection or less stable FAST_SCORE
        // int patchSize = 31;         // Size of the patch used by the oriented BRIEF descriptor.
        // int fastThreshold = 20;     // FAST threshold

        detector = cv::ORB::create();
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        // cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB; // Type of the extracted descriptor
        // int  	descriptor_size = 0;     // Size of the descriptor in bits. 0 -> Full size
        // int  	descriptor_channels = 3; // Number of channels in the descriptor (1, 2, 3)
        // float  	threshold = 0.001f;      // Detector response threshold to accept point
        // int  	nOctaves = 4;            // Maximum octave evolution of the image
        // int  	nOctaveLayers = 4;       // Default number of sublevels per scale level
        // cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type.

        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        // int nfeatures = 0;                // The number of best features to retain.
        // int nOctaveLayers = 3;            // The number of layers in each octave.
        // double contrastThreshold = 0.04;  // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        // double edgeThreshold = 10;        // The threshold used to filter out edge-like features
        // double sigma = 1.6;               // The sigma of the Gaussian applied to the input image at the octave #0

        detector = cv::xfeatures2d::SiftFeatureDetector::create();
    }

    // detect keypoints
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    duration = 1000 * t / 1.0;
}

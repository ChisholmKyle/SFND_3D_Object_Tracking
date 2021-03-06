
/* INCLUDES FOR THIS PROJECT */
#include <deque>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;


static void WriteHeaderToCsv(std::ofstream &csvFs)
{
    csvFs << "'Case'";
    csvFs << ","
          << "'Image'";

    csvFs << ","
          << "'Detector'";
    csvFs << ","
          << "'Descriptor'";

    csvFs << ","
          << "'cameraTTC (s)'";
    csvFs << ","
          << "'lidarTTC (s)'";
    csvFs << ","
          << "'Processing Duration (ms)'";

    csvFs << endl;
}


static void AppendToCsv(const MatchingResults &results, const MatchingParameters &parameters, std::ofstream &csvFs)
{
    csvFs << results.testCase;
    csvFs << "," << results.imageNumber;

    csvFs << "," << parameters.keypointDetectorType;
    csvFs << "," << parameters.keypointDescriptorType;

    csvFs << "," << results.cameraTTC;
    csvFs << "," << results.lidarTTC;

    csvFs << "," << results.processingTimeMs;

    csvFs << endl;
}

/* MAIN PROGRAM */
static void GenerateTest(const MatchingParameters &parameters, const int testCase, std::string &outputDirectory, std::ofstream &csvFs)
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = std::string(DATA_ROOT);

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1;
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;

    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;

    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (int imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {

        // new result for test case
        MatchingResults result;
        result.testCase = testCase;
        result.imageNumber = imgIndex;

        const std::string imgWritePrefix = outputDirectory + parameters.keypointDetectorType + "_" + parameters.keypointDescriptorType + "_img" + std::to_string(imgIndex);

        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);
        if (static_cast<int>(dataBuffer.size()) > dataBufferSize) {
            dataBuffer.pop_front();
        }

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        double detectObjectsDurationMs = 0.0;
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis, detectObjectsDurationMs);
        // detection with yolo is too time-consuming to consider when comparing
        // independent processing times for detector/descriptor
        // result.processingTimeMs += detectObjectsDurationMs;

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        double clusterDurationMs = 0.0;
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT, clusterDurationMs);
        result.processingTimeMs += clusterDurationMs;

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = parameters.keypointDetectorType;

        double detectorDurationMs = 0.0;
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, detectorDurationMs, bVis);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, detectorDurationMs, bVis);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, detectorDurationMs, bVis);
        }
        result.processingTimeMs += detectorDurationMs;

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        string descriptorType = parameters.keypointDescriptorType;

        double descriptorDurationMs = 0.0;
        try
        {
            descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, descriptorDurationMs);
        }
        catch (cv::Exception &e)
        {
            const char *err_msg = e.what();
            cout << std::endl
                 << "{ERROR} DETECTOR: " << detectorType << ", DESCRIPTOR: " << descriptorType << std::endl;
            std::cout << "exception caught: " << err_msg << std::endl;
            continue;
        }
        result.processingTimeMs += descriptorDurationMs;

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_" + parameters.matcherType;   // MAT_BF, MAT_FLANN
            string selectorType = "SEL_" + parameters.selectorType; // SEL_NN, SEL_KNN

            string matchDescriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            if (descriptorType.compare("SIFT") == 0)
            {
                matchDescriptorType = "DES_HOG";
            }

            double matchingDurationMs = 0.0;
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, matchDescriptorType, matcherType, selectorType, matchingDurationMs);
            result.processingTimeMs += matchingDurationMs;

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;


            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            double bbMatchDurationMs = 0.0;
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1), bbMatchDurationMs); // associate bounding boxes between current and previous frame using keypoint matches
            result.processingTimeMs += bbMatchDurationMs;
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB = nullptr, *currBB = nullptr;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                        break;
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                        break;
                    }
                }

                if (currBB == nullptr || prevBB == nullptr) continue;

                // compute TTC for current match
                if( currBB->lidarPoints.size() > 0u && prevBB->lidarPoints.size() > 0u ) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidarDurationMs = 0.0;
                    double ttcLidar;
                    std::vector<LidarPoint> outliers = computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar, ttcLidarDurationMs);
                    result.processingTimeMs += ttcLidarDurationMs;
                    //// EOF STUDENT ASSIGNMENT

                    // Visualize 3D objects
                    bVis = true;
                    if(parameters.showImage || parameters.saveImage)
                    {
                        // only show current BB
                        std::vector<BoundingBox> visBB = std::vector<BoundingBox>({*currBB});
                        show3DObjects(visBB, cv::Size(4.0, 20.0), cv::Size(2000, 2000), outliers, parameters, imgWritePrefix);
                    }
                    bVis = false;

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)

                    double ttcCameraDurationMs = 0.0;
                    double ttcCamera;

                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches, ttcCameraDurationMs);
                    result.processingTimeMs += ttcCameraDurationMs;

                    ttcCameraDurationMs = 0.0;
                    std::vector<cv::KeyPoint> ttcKeypoints = computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera, ttcCameraDurationMs);
                    result.processingTimeMs += ttcCameraDurationMs;

                    result.cameraTTC = ttcCamera;
                    result.lidarTTC = ttcLidar;

                    //// EOF STUDENT ASSIGNMENT

                    bVis = true;
                    if (parameters.showImage || parameters.saveImage)
                    {
                        cv::Mat lidarImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(lidarImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &lidarImg);
                        cv::rectangle(lidarImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

                        cv::Mat visImg = lidarImg.clone();
                        cv::drawKeypoints(lidarImg, ttcKeypoints, visImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        if (parameters.saveImage) {
                            cv::Point2i pt1 = cv::Point2i(400, 120);
                            cv::Point2i pt2 = cv::Point2i(850, 370);
                            cv::Rect cropRoi = cv::Rect(pt1, pt2);
                            cv::Mat croppedImg = visImg(cropRoi);
                            cv::imwrite(imgWritePrefix + "_cameraview.png", croppedImg);
                        }
                        if (parameters.showImage) {
                            cv::namedWindow(windowName, 4);
                            cv::imshow(windowName, visImg);
                            cout << "Press key to continue to next frame" << endl;
                            cv::waitKey(0);
                        }
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches

        }

        // write to CSV
        AppendToCsv(result, parameters, csvFs);

    } // eof loop over all images

}


/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES */

    // -------------------------------------------
    // Save select images setup:
    // -------------------------------------------

    // const bool showImage = false;
    // const bool saveImage = true;
    // std::vector<std::string> keypointDetectorList =
    //     {"FAST", "BRISK"};
    // std::vector<std::string> keypointDescriptorList =
    //     {"BRISK", "BRIEF", "SIFT", "ORB"};
    // const bool testAKAZE = false;

    // -------------------------------------------
    // No saved images, all keypoint matching data:
    // -------------------------------------------

    // show and/or save images
    const bool showImage = false;
    const bool saveImage = false;
    // test data tables
    std::vector<std::string> keypointDetectorList =
        {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    std::vector<std::string> keypointDescriptorList =
        {"BRISK", "BRIEF", "ORB", "FREAK", "SIFT"};
    const bool testAKAZE = true;

    // -------------------------------------------

    // timestamp for results
    time_t t = std::time(0);
    struct tm *now = std::localtime(&t);
    char timeString[80];
    std::strftime(timeString, 80, "%Y-%m-%d_%Hh%Mm%Ss", now);
    std::string resultsFolder = "results_" + std::string(timeString) + "/";

    // Creating results directory
    if (mkdir(resultsFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
        cerr << "Error :  " << strerror(errno) << endl;
        return EXIT_FAILURE;
    }

    std::vector<MatchingParameters> testParameters;
    for (const auto &detectorType : keypointDetectorList)
    {
        for (const auto &descriptorType : keypointDescriptorList)
        {
            MatchingParameters testParameter;
            testParameter.keypointDetectorType = detectorType;
            testParameter.keypointDescriptorType = descriptorType;
            testParameter.showImage = showImage;
            testParameter.saveImage = saveImage;

            // add to list
            testParameters.push_back(testParameter);
        }
    }
    if (testAKAZE) {
        // AKAZE descriptor requires AKAZE detector
        MatchingParameters testParameter;
        testParameter.keypointDetectorType = "AKAZE";
        testParameter.keypointDescriptorType = "AKAZE";
        testParameter.showImage = showImage;
        testParameter.saveImage = saveImage;

        testParameters.push_back(testParameter);
    }

    // open CSV file
    std::ofstream csvFs;
    std::string csvFilePath = resultsFolder + "/test_data.csv";
    csvFs.open(csvFilePath);
    WriteHeaderToCsv(csvFs);

    // generate results
    for (size_t testCase = 0; testCase < testParameters.size(); ++testCase)
    {
        GenerateTest(testParameters[testCase], testCase, resultsFolder, csvFs);
    }

    csvFs.close();
    return 0;
}

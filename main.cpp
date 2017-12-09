/*
 * 
 */
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


static void floodFillPostprocess(Mat&, const Scalar&);

int main(int argc, const char** argv) {

	// Contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Setup strings for input image locations
	char* gallery_file_location = "MediaGallery/";
	char* gallery_files[] = {
		"Gallery1.jpg",
		"Gallery2.jpg",
		"Gallery3.jpg",
		"Gallery4.jpg"
	};

	// Load images
	int number_of_gallery = sizeof(gallery_files) / sizeof(gallery_files[0]);  // 4 Images
	Mat* galleryImages = new Mat[number_of_gallery];  // Matrix of images
	for (int i = 0; i < number_of_gallery; i++) {
		string filename(gallery_file_location);
		filename.append(gallery_files[i]); // Creating path to image
		galleryImages[i] = imread(filename, -1);  // Read file into image matrix
		if (galleryImages[i].empty()) {
			cout << "Could not open " << galleryImages[i] << endl;
			return -1;
		}
	}
	cout << "Gallery images opened successfully\n";


	char* painting_file_location = "MediaPaintings/";
	char* painting_files[] = {
		"Painting1.jpg",
		"Painting2.jpg",
		"Painting3.jpg",
		"Painting4.jpg",
		"Painting5.jpg",
		"Painting6.jpg"
	};
	// Load images
	int number_of_painting = sizeof(painting_files) / sizeof(painting_files[0]);  // 4 Images
	Mat* paintingImages = new Mat[number_of_painting];  // Matrix of images
	for (int i = 0; i < number_of_painting; i++) {
		string filename(painting_file_location);
		filename.append(painting_files[i]); // Creating path to image
		paintingImages[i] = imread(filename, -1);  // Read file into image matrix
		if (paintingImages[i].empty()) {
			cout << "Could not open " << paintingImages[i] << endl;
			return -1;
		}
	}
	cout << "Painting images opened successfully\n";

	// REMOVE BEFORE FINAL SUBMISSION (JUST TO REDUCE COMPUTATION TIME)
	char* mean_file_location = "MediaMeanShift/";
	char* mean_files[] = {
		"MeanImage1.jpg",
		"MeanImage2.jpg",
		"MeanImage3.jpg",
		"MeanImage4.jpg"
	};
	Mat* mgalleryImages = new Mat[number_of_gallery];  // Matrix of images
	// Load images
	for (int i = 0; i < number_of_gallery; i++) {
		string filename(mean_file_location);
		filename.append(mean_files[i]); // Creating path to image
		mgalleryImages[i] = imread(filename, -1);  // Read file into image matrix
		if (mgalleryImages[i].empty()) {
			cout << "Could not open " << mgalleryImages[i] << endl;
			return -1;
		}
	}
	cout << "Mean Shift images opened successfully\n";


	for (int i = 0; i < number_of_gallery; i++) {

		// COMMENTED OUT TO SAVE COMPUTATION TIME (UNCOMMENT FOR FINAL SUBMISSION)
		//Mat mean_shift_image = Mat(galleryImages[i].size(), galleryImages[i].type());
		// Mean shift clustering/segmentation
		//pyrMeanShiftFiltering(galleryImages[i], mean_shift_image, 40, 30, 2);  // input, output, sp (spatial window radius), sr (colour window radius)
		//floodFillPostprocess(mean_shift_image, Scalar::all(2));

		int currentMaxColor = 0;
		Mat image = mgalleryImages[i].clone();
		vector<int> maxColorValue(3);
		vector<vector<vector<int>>> colorCounter(256, vector<vector<int>>(256, vector<int>(256, 0)));
		for (int y = 0;y < galleryImages[i].rows;y++)
		{
			for (int x = 0;x < galleryImages[i].cols;x++)
			{
				// get pixel
				Vec3b color = image.at<Vec3b>(Point(x, y));
				colorCounter[color[0]][color[1]][color[2]]++;
				if (colorCounter[color[0]][color[1]][color[2]] > currentMaxColor) {
					currentMaxColor = colorCounter[color[0]][color[1]][color[2]];
					maxColorValue[0] = color[0];
					maxColorValue[1] = color[1];
					maxColorValue[2] = color[2];
				}
			}
		}

		// Most common BGR value
		printf("%d, %d, %d\n", maxColorValue[0], maxColorValue[1], maxColorValue[2]);

		// Set all "wall" BGR values to zero, all others to 255

		Mat wallImage = mgalleryImages[i].clone();  // Copy by reference
		for (int y = 0;y < galleryImages[i].rows;y++) {
			for (int x = 0;x < galleryImages[i].cols;x++) {
				// get pixel
				Vec3b color = wallImage.at<Vec3b>(Point(x, y));
				if (color[0] < maxColorValue[0] + 15 && color[1] < maxColorValue[1] + 15 && color[2] < maxColorValue[2] + 15 &&
					color[0] > maxColorValue[0] - 15 && color[1] > maxColorValue[1] - 15 && color[2] > maxColorValue[2] - 15) {
					color[0] = 0;
					color[1] = 0;
					color[2] = 0;
				}
				else {
					color[0] = 255;  // color.val[0]
					color[1] = 255;
					color[2] = 255;
				}
				// set pixel
				wallImage.at<Vec3b>(Point(x, y)) = color;
			}
		}
		// Convert to greyscale image format
		Mat grey_image = Mat(galleryImages[i].size(), galleryImages[i].type());
		cvtColor(wallImage, grey_image, CV_BGR2GRAY);
		// Binary threshold
		Mat binary_image = Mat(galleryImages[i].size(), galleryImages[i].type());
		threshold(grey_image, binary_image, 128, 255, THRESH_BINARY);
		// Opening with 7x7 kernel
		Mat seven_by_seven_element(7, 7, CV_8U, Scalar(1));  // Seven by seven
		Mat opened_image = Mat(galleryImages[i].size(), galleryImages[i].type());
		morphologyEx(binary_image, opened_image, MORPH_OPEN, seven_by_seven_element);

		// Connected Components Analysis
		Mat binary_image_copy = opened_image.clone();
		findContours(binary_image_copy, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
		vector<vector<Point>> contours_poly(contours.size());  // MUST BE INITIALSIED AFTER CONTOURS ARE FOUND
		vector<Rect> boundRect(contours.size());  // MUST BE INITIALSIED AFTER CONTOURS ARE FOUND
		vector<Mat> paintingMask; // To contain a mask image for each painting found
		for (int contour_number = 0; (contour_number < (int)contours.size()); contour_number++)
		{
			approxPolyDP(Mat(contours[contour_number]), contours_poly[contour_number], 3, true);
			boundRect[contour_number] = boundingRect(Mat(contours_poly[contour_number]));
			float areaOfContour = contourArea(contours[contour_number]);

			if (areaOfContour > 15000.0) {  // Contour Area threshold
				int bottomYCoord = boundRect[contour_number].y + boundRect[contour_number].height;
				if (bottomYCoord < (galleryImages[i].rows - 2)) {  // Make sure it's not the ground
					//Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
					Scalar colour(0xFF, 0xFF, 0xFF);
					Mat contours_image = Mat::zeros(binary_image.size(), CV_8UC3);
					drawContours(contours_image, contours, contour_number, colour, CV_FILLED, 8, hierarchy);
					Mat temp = contours_image.clone();
					Mat temp_grey_image;
					cvtColor(temp, temp_grey_image, CV_BGR2GRAY);
					//// Binary threshold
					Mat temp_binary_image;
					threshold(temp_grey_image, temp_binary_image, 128, 255, THRESH_BINARY);
					temp_binary_image.convertTo(temp_binary_image, CV_8UC1);
					paintingMask.push_back(temp_binary_image);
					rectangle(galleryImages[i], boundRect[contour_number], colour, 2, 8, 0);
				}
			}
		}


		Mat hsv_image;
		cvtColor(galleryImages[i], hsv_image, COLOR_BGR2HSV);

		Mat hsv_painting1;
		cvtColor(paintingImages[0], hsv_painting1, COLOR_BGR2HSV);

		Mat hsv_painting2;
		cvtColor(paintingImages[1], hsv_painting2, COLOR_BGR2HSV);

		Mat hsv_painting3;
		cvtColor(paintingImages[2], hsv_painting3, COLOR_BGR2HSV);

		Mat hsv_painting4;
		cvtColor(paintingImages[3], hsv_painting4, COLOR_BGR2HSV);

		Mat hsv_painting5;
		cvtColor(paintingImages[4], hsv_painting5, COLOR_BGR2HSV);

		Mat hsv_painting6;
		cvtColor(paintingImages[5], hsv_painting6, COLOR_BGR2HSV);

		// Histogram setup variables
		int h_bins = 50; int s_bins = 60;
		int histSize[] = { h_bins, s_bins };
		// hue varies from 0 to 179, saturation from 0 to 255
		float h_ranges[] = { 0, 180 };
		float s_ranges[] = { 0, 256 };
		const float* ranges[] = { h_ranges, s_ranges };
		int channels[] = { 0, 1 };
		// ---------------------------
		
		MatND hist_base;
		MatND hist_painting1;
		MatND hist_painting2;
		MatND hist_painting3;
		MatND hist_painting4;
		MatND hist_painting5;
		MatND hist_painting6;

		calcHist(&hsv_painting1, 1, channels, Mat(), hist_painting1, 2, histSize, ranges, true, false);
		normalize(hist_painting1, hist_painting1, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&hsv_painting2, 1, channels, Mat(), hist_painting2, 2, histSize, ranges, true, false);
		normalize(hist_painting2, hist_painting2, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&hsv_painting3, 1, channels, Mat(), hist_painting3, 2, histSize, ranges, true, false);
		normalize(hist_painting3, hist_painting3, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&hsv_painting4, 1, channels, Mat(), hist_painting4, 2, histSize, ranges, true, false);
		normalize(hist_painting4, hist_painting4, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&hsv_painting5, 1, channels, Mat(), hist_painting5, 2, histSize, ranges, true, false);
		normalize(hist_painting5, hist_painting5, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&hsv_painting6, 1, channels, Mat(), hist_painting6, 2, histSize, ranges, true, false);
		normalize(hist_painting6, hist_painting6, 0, 1, NORM_MINMAX, -1, Mat());


		for (int j = 0; j < paintingMask.size(); j++) {
			calcHist(&hsv_image, 1, channels, paintingMask[j], hist_base, 2, histSize, ranges, true, false);  // Uses mask for each painting
			normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

			for (int k = 0; k < 4; k++) {
				int compare_method = k;
				double hist_comparison_p1 = compareHist(hist_base, hist_painting1, compare_method);
				double hist_comparison_p2 = compareHist(hist_base, hist_painting2, compare_method);
				double hist_comparison_p3 = compareHist(hist_base, hist_painting3, compare_method);
				double hist_comparison_p4 = compareHist(hist_base, hist_painting4, compare_method);
				double hist_comparison_p5 = compareHist(hist_base, hist_painting5, compare_method);
				double hist_comparison_p6 = compareHist(hist_base, hist_painting6, compare_method);
				printf("Gallery Painting [%d]. Gallery Number [%d]. Method [%d]:\nPainting 1: %f\nPainting 2: %f\nPainting 3: %f\nPainting 4: %f\nPainting 5: %f\nPainting 6: %f\n", j, i, k, 
					hist_comparison_p1, hist_comparison_p2, hist_comparison_p3, hist_comparison_p4, hist_comparison_p5, hist_comparison_p6);
			}
		}


		// Bitwise AND to get paiting from original image
		//for (int j = 0; j < paintingMask.size(); j++) {
		//	bitwise_and(paintingMask[j], galleryImages[i], paintingMask[j]);
		//}

		//You can check whether a contour with index i is inside another by checking if hierarchy[0,i,3] equals -1 or not. 
		//If it is different from -1, then your contour is inside another.

		for (int j = 0; j < paintingMask.size(); j++) {
			// Write image
			string outputName = "OutputImage";
			outputName.append(to_string(i + 1));
			outputName.append(to_string(j + 1));
			outputName.append(".jpg");
			imwrite(outputName, paintingMask[j]);  // image[i]
		}
	}
	printf("Press enter to finish");
	cin.ignore();
}


// This routine colors the regions
// Code taken from Open Source meanshift_segmentation.cpp (widely available online)
static void floodFillPostprocess(Mat& img, const Scalar& colorDiff = Scalar::all(1))
{
	CV_Assert(!img.empty());
	RNG rng = theRNG();
	Mat mask(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)
			{
				Scalar newVal(rng(256), rng(256), rng(256));
				floodFill(img, mask, Point(x, y), newVal, 0, colorDiff, colorDiff);
			}
		}
	}
	printf("Finished filling mean shift image");
}



	//for (int i = 0; i < number_of_painting; i++) {
	//	// Write image
	//	string outputName1 = "TestImage";
	//	outputName1.append(to_string(i + 1));
	//	outputName1.append(".jpg");
	//	imwrite(outputName1, paintingImages[i]);  // image[i]
	//}


	//// Contours
	//vector<vector<Point>> contours;
	//vector<Vec4i> hierarchy;

	//// Random variable for colour generation
	//RNG rng(1357);

	//// DICE coefficients
	//double dice_coefficient[8];

	//// Ground Truth Rectangles
	//vector<Rect> Notice1Ground(3);
	//vector<Rect> Notice2Ground(1);
	//vector<Rect> Notice3Ground(1);
	//vector<Rect> Notice4Ground(5);
	//vector<Rect> Notice5Ground(3);
	//vector<Rect> Notice6Ground(1);
	//vector<Rect> Notice7Ground(4);
	//vector<Rect> Notice8Ground(2);

	//// Notice 1
	//Notice1Ground[0] = Rect(Point(34, 17), Point(286, 107));
	//Notice1Ground[1] = Rect(Point(32, 117), Point(297, 223));
	//Notice1Ground[2] = Rect(Point(76, 234), Point(105, 252));
	//// Notice 2
	//Notice2Ground[0] = Rect(Point(47, 191), Point(224, 253));
	//// Notice 3
	//Notice3Ground[0] = Rect(Point(142, 121), Point(566, 392));
	//// Notice 4
	//Notice4Ground[0] = Rect(Point(157, 72), Point(378, 134));
	//Notice4Ground[1] = Rect(Point(392, 89), Point(448, 132));
	//Notice4Ground[2] = Rect(Point(405, 138), Point(442, 152));
	//Notice4Ground[3] = Rect(Point(80, 157), Point(410, 245));
	//Notice4Ground[4] = Rect(Point(82, 258), Point(372, 322));
	//// Notice 5
	//Notice5Ground[0] = Rect(Point(112, 73), Point(598, 170));
	//Notice5Ground[1] = Rect(Point(108, 178), Point(549, 256));
	//Notice5Ground[2] = Rect(Point(107, 264), Point(522, 352));
	//// Notice 6
	//Notice6Ground[0] = Rect(Point(91, 54), Point(446, 227));
	//// Notice 7
	//Notice7Ground[0] = Rect(Point(64, 64), Point(476, 268));
	//Notice7Ground[1] = Rect(Point(529, 126), Point(611, 188));
	//Notice7Ground[2] = Rect(Point(545, 192), Point(603, 211));
	//Notice7Ground[3] = Rect(Point(210, 305), Point(595, 384));
	//// Notice 8
	//Notice8Ground[0] = Rect(Point(158, 90), Point(768, 161));
	//Notice8Ground[1] = Rect(Point(114, 174), Point(800, 279));

	//// Ground truth to store all rectangles and their corresponding images
	//vector<vector<Rect>> groundTruth;
	//groundTruth.push_back(Notice1Ground);
	//groundTruth.push_back(Notice2Ground);
	//groundTruth.push_back(Notice3Ground);
	//groundTruth.push_back(Notice4Ground);
	//groundTruth.push_back(Notice5Ground);
	//groundTruth.push_back(Notice6Ground);
	//groundTruth.push_back(Notice7Ground);
	//groundTruth.push_back(Notice8Ground);



	//// Loop over all images
	//for(int i = 0; i < number_of_images; i++){
	//	//--------------------------------------------------
	//	// K Means implementation from "A Practical Introduction to Computer Vision with OpenCV"
	//	// by Kenneth Dawson - Howe © Wiley & Sons Inc. 2014.  All rights reserved.
	//	//--------------------------------------------------
	//	Mat samples(image[i].rows*image[i].cols, 3, CV_32F);
	//	float* sample = samples.ptr<float>(0);
	//	for (int row = 0; row < image[i].rows; row++)
	//		for (int col = 0; col < image[i].cols; col++)
	//			for (int channel = 0; channel < 3; channel++)
	//				samples.at<float>(row*image[i].cols + col, channel) =
	//				(uchar)image[i].at<Vec3b>(row, col)[channel];
	//	// Apply k-means clustering, determining the cluster
	//	// centres and a label for each pixel.
	//	int k = 3;  // number of clusters (ASSUMPTION) 
	//	Mat labels, centres;
	//	kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER |
	//		CV_TERMCRIT_EPS, 0.0001, 10000), 1,
	//		KMEANS_PP_CENTERS, centres);
	//	// Use centres and label to populate result image
	//	Mat k_means_image = Mat(image[i].size(), image[i].type());
	//	for (int row = 0; row < image[i].rows; row++)
	//		for (int col = 0; col < image[i].cols; col++)
	//			for (int channel = 0; channel < 3; channel++)
	//				k_means_image.at<Vec3b>(row, col)[channel] =
	//				(uchar)centres.at<float>(*(labels.ptr<int>(
	//					row*image[i].cols + col)), channel);
	//	//--------------------------------------------------
	//	//  End of K-Means Implementation
	//	//--------------------------------------------------

	//	// Create a greyscale image 
	//	Mat grey_image = Mat(image[i].size(), image[i].type());
	//	cvtColor(k_means_image, grey_image, CV_BGR2GRAY); 

	//	// Binary threshold the greyscale image (ASSUMPTION MADE THAT SIGNS ARE HIGH CONTRAST)
	//	Mat binary_image = Mat(image[i].size(), image[i].type());
	//	threshold(grey_image, binary_image, 128, 255, THRESH_BINARY);

	//	// Connected Components Analysis
	//	Mat binary_image_copy = binary_image.clone();
	//	findContours(binary_image_copy, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	//	Mat contours_image = Mat::zeros(binary_image.size(), CV_8UC3);
	//	vector<vector<Point> > contours_poly(contours.size());  // MUST BE INITIALSIED AFTER CONTOURS ARE FOUND
	//	vector<Rect> boundRect(contours.size());  // MUST BE INITIALSIED AFTER CONTOURS ARE FOUND

	//	// create bounding rectangle for each contour
	//	for (int cont_num = 0; (cont_num<(int)contours.size()); cont_num++)
	//	{
	//		approxPolyDP(Mat(contours[cont_num]), contours_poly[cont_num], 3, true);
	//		boundRect[cont_num] = boundingRect(Mat(contours_poly[cont_num]));
	//	}

	//	vector<int> sameLineCount(contours.size(), 0);  // Count of all other contours at a similar height and area to this one
	//	// Iterating through for each contour
	//	for (int cont_num = 0; (cont_num<(int)contours.size()); cont_num++)
	//	{
	//		double areaOfContour = contourArea(contours[cont_num]);
	//		// Iterate through all the other contours and see if the bottom left corner of this one is on a similar
	//		// height to the currently "selected" contour

	//		// Stores the number of other contours found around this one
	//		int blCornery_curr = boundRect[cont_num].y + boundRect[cont_num].height;  // bottom left corner y position of current rectangle of interest
	//		int area_curr = boundRect[cont_num].width * boundRect[cont_num].height; // area of current rectangle of interest
	//		// compare boundRect[cont_num] to all the other bounding rectangles
	//		for (int comparison = 0; (comparison < (int)contours.size()); comparison++) {
	//			if (comparison == cont_num) {
	//				;  // Don't want to compare with self
	//			}
	//			else {
	//				int blCornery_comp = boundRect[comparison].y + boundRect[comparison].height;  // bottom left corner y position
	//				int area_comp = boundRect[comparison].width * boundRect[comparison].height;  // bottom left corner x position
	//				// Horizontal detection. Check if within +/-40 pixels vertically with other regions 
	//				if (blCornery_comp < blCornery_curr + 40 && blCornery_comp > blCornery_curr - 40) {  		
	//					// If either region is not double the size of the other
	//					if (area_comp < area_curr*2 && area_comp > area_curr/2) {
	//						sameLineCount[comparison]++;
	//					}
	//				}
	//			}	
	//		}
	//	}

	//	// Dice coefficient parameters
	//	double dice_overlap = 0;
	//	double dice_ground_area = 0;
	//	double dice_found_area = 0;

	//	for (int cont_num = 0; (cont_num < (int)contours.size()); cont_num++)
	//	{
	//		// Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
	//		Scalar colour(0x00, 0xFF, 0x00);
	//		double areaOfContour = contourArea(contours[cont_num]);
	//		// Minimum area of acceptable contour 
	//		if (areaOfContour > 30 && areaOfContour < 5000 && sameLineCount[cont_num] > 3) {  
	//			// drawContours(contours_image, contours, cont_num, colour, CV_FILLED, 8, hierarchy);  // Draws contour
	//			rectangle(image[i], boundRect[cont_num], colour, 2, 8, 0);  // Draw bounding rectangle
	//			dice_found_area = dice_found_area + boundRect[cont_num].area();  // Overall area of solution (for DICE calculation)
	//		}
	//	}

	//	// Draw bounding rectangles for ground truth 
	//	for (int j = 0; j < groundTruth[i].size(); j++){
	//		//Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
	//		Scalar colour(0xFF, 0x00, 0xFF);
	//		rectangle(image[i], groundTruth[i][j], colour, 2, 8, 0);  
	//	}

	//	// Calculate overlap area for DICE
	//	for (int j = 0; j < groundTruth[i].size(); j++) {
	//		dice_ground_area = dice_ground_area + groundTruth[i][j].area();
	//		for (int k = 0; k < (int)contours.size(); k++) {
	//			if ((boundRect[k] & groundTruth[i][j]).area() > 0) {
	//				double areaOfContour = contourArea(contours[k]);
	//				if (areaOfContour > 30 && areaOfContour < 5000 && sameLineCount[k] > 3) {
	//					dice_overlap = dice_overlap + (boundRect[k] & groundTruth[i][j]).area();
	//				}
	//			}
	//		}
	//	}

	//	// Calculate DICE coefficient
	//	dice_coefficient[i] = 2 * dice_overlap / (dice_ground_area + dice_found_area);

	//	// Write image
	//	string kMeansName = "OutputImage";
	//	kMeansName.append(to_string(i + 1));
	//	kMeansName.append(".jpg");
	//	imwrite(kMeansName, image[i]);  // image[i]

	//	cout << "--------------------\nImage " << i + 1 << " has been processed\n--------------------\n";
	//	
	//	// Print DICE parameters and coefficient
	//	printf("area covered by implementation: %f\n", dice_found_area);
	//	printf("area covered by ground truth: %f\n", dice_ground_area);
	//	printf("area of overlap: %f\n", dice_overlap);
	//	printf("DICE coefficient: %f\n\n", dice_coefficient[i]);

	//	// Deallocate vector memory
	//	boundRect.clear();
	//	vector<Rect>().swap(boundRect);
	//	sameLineCount.clear();
	//	vector<int>().swap(sameLineCount);
	//}

	//double dice_average = 0;
	//for (int i = 0; i < 8; i++) {
	//	dice_average = dice_average + dice_coefficient[i];
	//}
	//
	//// Overall DICE average value
	//dice_average = dice_average / 8;
	//printf("--------------------\nOverall DICE coefficient average: %f\n--------------------\n", dice_average);

	//cout << "Type something and press enter to end: ";
	//cin.ignore();



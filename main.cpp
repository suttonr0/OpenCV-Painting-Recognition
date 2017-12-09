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
					Scalar foundColour(0x00, 0xFF, 0x00);
					drawContours(galleryImages[i], contours, contour_number, foundColour, 3, 8, hierarchy);
					//rectangle(galleryImages[i], boundRect[contour_number], colour, 2, 8, 0);
				}
			}
		}


		Mat hsv_image;
		cvtColor(galleryImages[i], hsv_image, COLOR_BGR2HSV);
		Mat hsv_painting[6];
		for (int j = 0; j < 6; j++) {
			cvtColor(paintingImages[j], hsv_painting[j], COLOR_BGR2HSV);
		}
		
		// Histogram setup variables
		int h_bins = 100; int s_bins = 120;  // Hue and saturation bins
		int histSize[] = { h_bins, s_bins };
		// hue varies from 0 to 179, saturation from 0 to 255
		float h_ranges[] = { 0, 180 };
		float s_ranges[] = { 0, 256 };
		const float* ranges[] = { h_ranges, s_ranges };
		int channels[] = { 0, 1 };
		// ---------------------------
		
		MatND hist_base;
		MatND hist_painting[6];

		for (int j = 0; j < 6; j++) {
			calcHist(&hsv_painting[j], 1, channels, Mat(), hist_painting[j], 2, histSize, ranges, true, false);
			normalize(hist_painting[j], hist_painting[j], 0, 1, NORM_MINMAX, -1, Mat());
		}
	
		int* maxCorrelation = new int[paintingMask.size()]; // Will store the painting index for the max correlation
		
		for (int j = 0; j < paintingMask.size(); j++) {
			calcHist(&hsv_image, 1, channels, paintingMask[j], hist_base, 2, histSize, ranges, true, false);  // Uses mask for each painting
			normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
			int compare_method = 0;  // Correlation method
			double hist_comparison[6];
			printf("Gallery Painting [%d]. Gallery Number [%d]:\n", j, i);
			double currentMaxCorr = 0;  // initialise current max correlation value to zero
			maxCorrelation[j] = 0; // index of painting with max correlation initialised to zero
			for (int k = 0; k < 6; k++) {
				hist_comparison[k] = compareHist(hist_base, hist_painting[k], compare_method);
				if (hist_comparison[k] > currentMaxCorr) {
					currentMaxCorr = hist_comparison[k];
					maxCorrelation[j] = k;
				}
				printf("Painting [%d]: %f\n", k, hist_comparison[k]);
			}
			printf("Max correlation found for painting %i\n", maxCorrelation[j]);
		}

		int correlationCounter = 0;  // A counter to iterate through the number of paintings for the current gallery image
		for (int contour_number = 0; (contour_number < (int)contours.size()); contour_number++)
		{
			approxPolyDP(Mat(contours[contour_number]), contours_poly[contour_number], 3, true);
			boundRect[contour_number] = boundingRect(Mat(contours_poly[contour_number]));
			float areaOfContour = contourArea(contours[contour_number]);

			if (areaOfContour > 15000.0) {  // Contour Area threshold
				int bottomYCoord = boundRect[contour_number].y + boundRect[contour_number].height;
				if (bottomYCoord < (galleryImages[i].rows - 2)) {  // Make sure it's not the ground
					Scalar foundColour(0x00, 0xFF, 0x00);
					string foundPaintingString = "Painting ";
					foundPaintingString.append(to_string(maxCorrelation[correlationCounter] + 1));
					correlationCounter++;
					putText(galleryImages[i], foundPaintingString, Point(boundRect[contour_number].x, boundRect[contour_number].y), FONT_HERSHEY_SIMPLEX, 1, foundColour, 2, 8, false);
				}
			}
		}

		delete[] maxCorrelation;

		// Bitwise AND to get paiting from original image
		//for (int j = 0; j < paintingMask.size(); j++) {
		//	bitwise_and(paintingMask[j], galleryImages[i], paintingMask[j]);
		//}

		// Draw ground truth
		Scalar groundColour(0xFF, 0x00, 0x00);
		if (i == 0) {
			// Painting 2
			line(galleryImages[i], Point(212, 261), Point(445, 255), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(445, 255), Point(428, 725), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(428, 725), Point(198, 673), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(198, 673), Point(212, 261), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 2", Point(198, 673), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);
			
			// Painting 1
			line(galleryImages[i], Point(686, 377), Point(1050, 361), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1050, 361), Point(1048, 705), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1048, 705), Point(686, 652), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(686, 652), Point(686, 377), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 1", Point(686, 652), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);
		}

		else if (i == 1) {
			// Painting 3
			line(galleryImages[i], Point(252, 279), Point(691, 336), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(691, 336), Point(695, 662), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(695, 662), Point(258, 758), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(258, 758), Point(252, 279), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 3", Point(258, 788), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);

			// Painting 2
			line(galleryImages[i], Point(897, 173), Point(1063, 234), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1063, 234), Point(1079, 672), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1079, 672), Point(917, 739), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(917, 739), Point(897, 173), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 2", Point(917, 769), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);

			// Painting 1
			line(galleryImages[i], Point(1174, 388), Point(1221, 395), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1221, 395), Point(1216, 544), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1216, 544), Point(1168, 555), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1168, 555), Point(1174, 388), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 1", Point(1118, 585), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);

		}

		else if (i == 2) {
			// Painting 4
			line(galleryImages[i], Point(68, 329), Point(350, 337), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(350, 337), Point(351, 545), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(351, 545), Point(75, 558), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(75, 558), Point(68, 329), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 4", Point(75, 588), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);

			// Painting 5
			line(galleryImages[i], Point(629, 346), Point(877, 350), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(877, 350), Point(873, 517), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(873, 517), Point(627, 530), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(627, 530), Point(629, 346), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 5", Point(627, 560), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);

			// Painting 6
			line(galleryImages[i], Point(1057, 370), Point(1187, 374), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1187, 374), Point(1182, 487), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1182, 487), Point(1053, 493), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1053, 493), Point(1057, 370), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 6", Point(1053, 523), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);
		}

		else if (i == 3) {
			// Painting 4
			line(galleryImages[i], Point(176, 348), Point(298, 347), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(298, 347), Point(307, 481), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(307, 481), Point(184, 475), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(184, 475), Point(176, 348), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 4", Point(184, 505), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);

			// Painting 5
			line(galleryImages[i], Point(469, 343), Point(690, 338), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(690, 338), Point(692, 495), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(692, 495), Point(472, 487), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(472, 487), Point(469, 343), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 5", Point(472, 517), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);
			
			// Painting 6
			line(galleryImages[i], Point(924, 349), Point(1161, 344), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1161, 344), Point(1156, 495), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(1156, 495), Point(924, 488), groundColour, 3, 8, 0);
			line(galleryImages[i], Point(924, 488), Point(924, 349), groundColour, 3, 8, 0);
			putText(galleryImages[i], "Painting 6", Point(924, 518), FONT_HERSHEY_SIMPLEX, 1, groundColour, 2, 8, false);
		}

		// Write image
		string outputName = "OutputImage";
		outputName.append(to_string(i + 1));
		outputName.append(".jpg");
		imwrite(outputName, galleryImages[i]);  // image[i]

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



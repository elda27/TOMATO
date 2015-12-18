#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <vector>
#include <Windows.h>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace cv;
void colorExtraction(cv::Mat* src, cv::Mat* dst,
	int code,
	int ch1Lower, int ch1Upper,
	int ch2Lower, int ch2Upper,
	int ch3Lower, int ch3Upper
	);

int main(void)
{
	VideoCapture video("C:/Users/idlab/Documents/Visual Studio 2013/Projects/SP360_param/SP360_param/SP360.MP4");
	namedWindow("frame", CV_WINDOW_AUTOSIZE);
	namedWindow("dst", CV_WINDOW_AUTOSIZE);
	namedWindow("hough", CV_WINDOW_AUTOSIZE);
	int low_H = 240,
		low_S = 7,
		low_V = 160,
		high_H = 237,
		high_S = 5,
		high_V = 255;

	int tomatos_prev = 0, tomatos_next = 0, alltomatos = 0;


	createTrackbar("low_H", "dst", &low_H, 255);
	createTrackbar("low_S", "dst", &low_S, 255);
	createTrackbar("low_V", "dst", &low_V, 255);
	createTrackbar("high_H", "dst", &high_H, 255);
	createTrackbar("high_S", "dst", &high_S, 255);
	createTrackbar("high_V", "dst", &high_V, 255);

	

	Mat frame, src, dst, hough;
	video >> src;
	Mat mask(src.size(), src.type(), Scalar::all(0));
	circle(mask, Point(mask.rows / 2, mask.cols / 2), mask.rows / 2 - 10, cv::Scalar::all(255), -1);

	while (!GetAsyncKeyState(VK_ESCAPE))
	{
		if (waitKey(1) >= 0)
		{
			video.set(CV_CAP_PROP_POS_MSEC, video.get(CV_CAP_PROP_POS_MSEC) + 10000);
		}

		video >> src;
		if (src.empty() || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1)		break;
		src.copyTo(frame, mask);

		cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
		colorExtraction(&frame, &dst, CV_BGR2HSV, low_H, low_S, low_V, high_H, high_S, high_V);
		cvtColor(dst, hough, CV_BGR2GRAY);
		morphologyEx(hough, hough, MORPH_CLOSE, Mat(), Point(-1,-1), 5);
		// 平滑化を行います．これがないと誤検出が起こりやすくなります．
		GaussianBlur(hough, hough, Size(9, 9), 2, 2);
		Mat binary, labelImage(frame.size(), CV_32S);
		threshold(hough, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
		vector<Point2f> centroids;
		vector<ConnectedComponentsTypes> stats;
		//int nLabels = connectedComponentsWithStats(binary, labelImage, stats, centroids);
		
		int nLabels = connectedComponents(binary, labelImage);
		vector<Vec3b> colors(nLabels);
		colors[0] = cv::Vec3b(0, 0, 0);
		for (int label = 1; label < nLabels; ++label)
		{
			colors[label] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
		}

		// ラベリング結果の描画
		for (int y = 0; y < frame.rows; ++y)
		{
			for (int x = 0; x < frame.cols; ++x)
			{
				int label = labelImage.at<int>(y, x);
				if (label == 0) continue;
				cv::Vec3b &pixel = frame.at<cv::Vec3b>(y, x);
				pixel = colors[label];
				
			}
		}

		tomatos_next = 0;
		for (int x = 1; x < frame.cols; x++)
		{
			int label_prev = labelImage.at<int>(frame.rows / 2, x - 1);
			int label_next = labelImage.at<int>(frame.rows / 2, x);
			if (label_prev != label_next)
			{
				tomatos_next++;
			}
		}

		tomatos_next /= 2;

		if (tomatos_prev > tomatos_next)
		{
			alltomatos += tomatos_prev - tomatos_next;
			cout << "tomatos = " << alltomatos << endl;
		}

		tomatos_prev = tomatos_next;

		imshow("frame", frame);
		imshow("dst", dst);
		imshow("hough", binary);

	}
	destroyWindow("frame");
	destroyWindow("dst");
	return 0;
}

void colorExtraction(cv::Mat* src, cv::Mat* dst,
	int code,
	int ch1Lower, int ch1Upper,
	int ch2Lower, int ch2Upper,
	int ch3Lower, int ch3Upper
	)
{
	cv::Mat colorImage;
	int lower[3];
	int upper[3];

	cv::Mat lut = cv::Mat(256, 1, CV_8UC3);

	cv::cvtColor(*src, colorImage, code);

	lower[0] = ch1Lower;
	lower[1] = ch2Lower;
	lower[2] = ch3Lower;

	upper[0] = ch1Upper;
	upper[1] = ch2Upper;
	upper[2] = ch3Upper;

	for (int i = 0; i < 256; i++){
		for (int k = 0; k < 3; k++){
			if (lower[k] <= upper[k]){
				if ((lower[k] <= i) && (i <= upper[k])){
					lut.data[i*lut.step + k] = 255;
				}
				else{
					lut.data[i*lut.step + k] = 0;
				}
			}
			else{
				if ((i <= upper[k]) || (lower[k] <= i)){
					lut.data[i*lut.step + k] = 255;
				}
				else{
					lut.data[i*lut.step + k] = 0;
				}
			}
		}
	}

	//LUTを使用して二値化
	cv::LUT(colorImage, lut, colorImage);

	//Channel毎に分解
	std::vector<cv::Mat> planes;
	cv::split(colorImage, planes);

	//マスクを作成
	cv::Mat maskImage;
	cv::bitwise_and(planes[0], planes[1], maskImage);
	cv::bitwise_and(maskImage, planes[2], maskImage);

	//出力
	cv::Mat maskedImage;
	src->copyTo(maskedImage, maskImage);
	*dst = maskedImage;
}
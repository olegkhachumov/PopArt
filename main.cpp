#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

#define MAX_IMAGE_SIZE 4088
#define MAX_WINDOW_SIZE  700
#define MAX_HALFTONE_SIZE 50


struct ContextPopArt {
	Mat src_image;			//original image
	Mat dst;				//result image
	Mat src_grayscale;		//grayscale image
	Mat src_bg;				//sharpen grayscale image
	Mat background_lut;		//lut for background
	Mat halftone_lut;		//lut for halftone effect
	int effect_amount;		//effect amount
	int halftone_radius;	//halftone radius

	~ContextPopArt() {
		src_image.release();
		dst.release();
		src_grayscale.release();
		src_bg.release();
		background_lut.release();
		halftone_lut.release();
	}
};

unsigned char colormap_background[3][256] = {
	{ 158, 158, 158, 158, 158, 161, 161, 158, 150, 148, 146, 147, 145, 141, 138, 134, 132, 128, 125, 122, 120, 117, 113, 110, 108, 104, 102, 99, 95, 93, 90, 87, 85, 81, 77, 76, 72, 67, 66, 63, 60, 58, 54, 52, 49, 45, 43, 40, 37, 35, 32, 29, 26, 24, 22, 22, 22, 21, 22, 22, 23, 23, 25, 26, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 26, 24, 26, 26, 26, 26, 26, 26, 25, 24, 26, 26, 26, 26, 26, 26, 25, 26, 24, 27, 25, 26, 26, 25, 25, 26, 24, 25, 24, 25, 25, 24, 25, 22, 24, 25, 25, 23, 23, 23, 23, 24, 23, 22, 23, 22, 21, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 19, 19, 19, 19, 20, 20, 19, 19, 18, 18, 18, 19, 19, 19, 19, 18, 18, 18, 18, 17, 17, 17, 18, 19, 20, 20, 19, 19, 18, 18, 17, 17, 17, 18, 18, 19, 19, 19, 18, 18, 18, 18, 17, 17, 17, 17, 16, 16, 16, 17, 17, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 },
{ 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 17, 26, 33, 39, 45, 51, 57, 62, 68, 73, 79, 83, 89, 94, 100, 105, 110, 115, 121, 126, 131, 137, 142, 146, 152, 157, 163, 167, 173, 177, 182, 184, 185, 186, 187, 188, 188, 189, 190, 191, 191, 192, 193, 194, 195, 195, 196, 196, 197, 197, 198, 198, 199, 200, 201, 202, 203, 203, 204, 205, 205, 205, 206, 206, 206, 207, 208, 209, 210, 211, 212, 213, 213, 214, 214, 215, 215, 216, 216, 217, 218, 219, 220, 221, 221, 222, 222, 222, 223, 224, 224, 225, 226, 226, 227, 227, 227, 228, 228, 229, 230, 231, 232, 232, 232, 233, 233, 233, 233, 234, 234, 235, 236, 236, 237, 238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 248, 249, 249, 249, 249, 249, 249, 250, 251, 252, 252, 253, 253, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255 },
{ 23, 23, 23, 23, 23, 22, 24, 24, 29, 30, 32, 34, 35, 37, 41, 44, 48, 49, 53, 56, 60, 63, 66, 69, 73, 77, 80, 85, 88, 93, 98, 101, 106, 109, 115, 119, 124, 129, 133, 137, 142, 147, 152, 156, 162, 166, 172, 176, 181, 187, 190, 195, 200, 205, 209, 214, 218, 222, 227, 232, 236, 240, 245, 248, 252, 254, 255, 254, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 254, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 254, 255, 255, 255, 254, 255, 254, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 254, 254, 253, 253, 253, 254, 254, 254, 254, 255, 255, 255, 255, 254, 254, 254, 254, 254, 254, 254, 254, 253, 253, 253, 254, 255, 255, 255, 254, 254, 254, 254, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254, 255, 255, 255, 255, 254, 254, 254, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 252, 252, 252, 253, 253, 254, 254, 254, 254, 254, 254, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254 }
};

unsigned char colormap_halftone[3][256] = {
	{ 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 159, 161, 161, 161, 162, 162, 163, 163, 163, 162, 162, 163, 162, 162, 161, 162, 162, 160, 160, 160, 162, 162, 162, 160, 160, 160, 158, 158, 156, 156, 156, 156, 156, 156, 157, 156, 156, 155, 154, 154, 154, 152, 152, 152, 150, 150, 150, 150, 148, 149, 149, 148, 147, 147, 146, 146, 146, 147, 146, 146, 146, 143, 143, 144, 143, 143, 143, 143, 141, 139, 139, 139, 138, 138, 138, 138, 138, 136, 135, 135, 135, 133, 133, 133, 132, 132, 132, 132, 131, 129, 129, 128, 128, 128, 125, 124, 124, 124, 123, 123, 121, 120, 120, 120, 120, 120, 119, 119, 118, 116, 116, 115, 116, 116, 115, 111, 111, 110, 110, 110, 108, 102, 102, 95, 90, 90, 85, 85, 82, 76, 76, 70, 65, 65, 61, 56, 56, 49, 44, 44, 40, 44, 44, 33, 26, 26, 22, 22, 25, 21, 21, 22, 24, 24, 24, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 25, 26, 26, 26, 26, 26, 26, 24, 24, 26, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 25, 26, 26, 24, 25, 25, 25, 25, 25, 25, 23, 23, 24, 25, 25, 24, 24, 24, 23, 23, 23, 23, 23, 22, 23, 23, 22, 22, 22, 23, 22, 22, 22, 21, 21, 20, 20, 20, 20, 20, 18, 20, 20, 19, 16, 16, 16, 16, 16 },
{ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 25, 1, 1, 20, 29, 29, 40, 48, 48, 58, 63, 63, 73, 79, 79, 89, 92, 92, 99, 109, 109, 114, 114, 106, 129, 129, 135, 127, 127, 149, 154, 154, 160, 169, 169, 175, 178, 178, 182, 188, 188, 197, 197, 201, 205, 205, 210, 214, 214, 216,  255, 255, 255, 255, 255 },
{ 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 22, 21, 21, 21, 21, 21, 22, 22, 21, 22, 22, 22, 22, 22, 24, 24, 24, 23, 23, 23, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 26, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 32, 32, 32, 34, 33, 33, 35, 36, 36, 35, 36, 36, 36, 36, 38, 39, 39, 40, 41, 41, 41, 41, 41, 42, 44, 44, 43, 45, 45, 46, 47, 47, 48, 48, 50, 49, 49, 49, 49, 49, 53, 54, 54, 54, 56, 56, 56, 57, 57, 59, 58, 58, 61, 61, 62, 63, 63, 64, 64, 64, 64, 68, 68, 69, 70, 70, 74, 81, 81, 88, 97, 97, 105, 105, 109, 118, 118, 127, 136, 136, 140, 150, 150, 158, 167, 167, 178, 170, 170, 191, 200, 200, 209, 209, 203, 222, 222, 231, 239, 239, 243, 251, 251, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 254, 255, 255, 255, 255, 255, 255, 253, 253, 254, 254, 253, 255, 255, 253, 253, 253, 253,  255, 255, 255, 255, 255 }
};


Mat createLUT(unsigned char(*colormap)[256]) {
	Mat channels[] = { Mat(256,1, CV_8U, colormap[0]), Mat(256,1, CV_8U, colormap[1]), Mat(256,1, CV_8U, colormap[2]) };
	Mat lut; // Create a lookup table
	merge(channels, 3, lut);
	return lut;
}


void applyCustomColorMap(Mat const & im_gray, Mat & im_color, Mat const & lut)
{
	/*
		Apply colormap lut to im_gray.
	*/
	Mat temp;
	cvtColor(im_gray.clone(), temp, COLOR_GRAY2BGR);
	LUT(temp, lut, im_color);

}


void window_setup(std::string name_window, int rows, int cols)
{
	namedWindow(name_window, CV_WINDOW_NORMAL);
	double coef = (double)max(rows, cols) / MAX_WINDOW_SIZE;
	int new_height = rows / coef;
	int new_width = cols / coef;
	resizeWindow(name_window, new_width, new_height + 70);
}


Mat sharp(Mat const &image, const int &radius) {
	/*
		Creation sharpen image.
	*/
	if (radius == 0)
		return image;
	Mat dst;
	blur(image, dst, Size(radius, radius), Point(-1, -1), 2);	//blurring image
	addWeighted(image, 2, dst, -1, 0, dst);						//difference between src image and blurred image
	return dst;
}

Mat halftone_mask(Mat const &im_gray, const int &halftone_size) {
	/*
		Creation of a grayscale mask based on an image in grayscale.
		The radius will be greater than the darker pixels in the center of the circle.
	*/
	Mat mask = Mat::zeros(im_gray.rows, im_gray.cols, CV_8UC1);		//
	int diameter = (static_cast<double>(max(im_gray.rows, im_gray.cols)) / 512)* halftone_size;
	Mat_<uchar> img = im_gray;
	if (diameter < 3)
		diameter = 3;
	int radius = ((double)(diameter + 1) / 2);	//max radius of circle

	double temp_radius = 0;
	for (int i = radius - 1; i < im_gray.rows; i += diameter) {
		for (int j = radius - 1; j < im_gray.cols; j += diameter)
		{
			temp_radius = ((double)abs(255.0 - img(i, j)*0.85)) / 256 * (radius);		//circle radius for the particular color
			circle(mask, Point(j, i), temp_radius, Scalar(255, 255, 255), -1, 8, 0);	//drawing white circle on mask 
			circle(mask, Point(j, i), temp_radius, Scalar(255, 255, 255), -1, 4, 0);	//another one with different borderType
		}
	}
	return mask;
}


void PopArt(void *data) {
	/*
		Creating Pop Art effect as same as befunky.com.
	*/
	ContextPopArt *context = reinterpret_cast<ContextPopArt *>(data);

	Mat mask = halftone_mask(context->src_grayscale, context->halftone_radius);
	Mat temp;
	context->src_bg.copyTo(temp, mask);
	applyCustomColorMap(temp, temp, context->halftone_lut);
	applyCustomColorMap(context->src_bg, context->dst, context->background_lut);
	temp.copyTo(context->dst, mask);
	blur(context->dst, context->dst, Size(2, 2), Point(-1, -1), 2);
	mask.release();
	temp.release();
}

void updateAmount(int, void *data) {

	ContextPopArt *context = reinterpret_cast<ContextPopArt *>(data);
	Mat dst = context->dst.clone();
	double ratio = context->effect_amount * 1.0 / 100;					// effect amount in percent
	dst = (1 - ratio) * context->src_image + ratio * context->dst;		// superposition of src_image and dst_image
	imshow("Pop Art", dst);
}

void updatePopArt(int, void *data) {
	PopArt(data);			//apply PopArt for new halftone size
	updateAmount(0, data);	//update the displayed image with the old amount
}

void resize_image(Mat const & src, Mat & dst) {
	double coef = (double)max(src.rows, src.cols) / MAX_IMAGE_SIZE;
	resize(src, dst, Size(src.cols / coef, src.rows / coef));
};


ContextPopArt setup(Mat & src) {
	int temp = min(2048, max(src.rows, src.cols));
	int coef = pow(2, temp / 1024);							//blur ratio for sharp

	if (max(src.rows, src.cols) > MAX_IMAGE_SIZE) {			//resize image if its size large then 4088x4088 pix
		resize_image(src, src);
	}

	Mat dst(src.size(), src.type());						//creating result image
	Mat src_grayscale;										//creating grayscale image
	cvtColor(src, src_grayscale, COLOR_BGR2GRAY);

	Mat src_bg;												//creating sharpen grayscale background image 
	src_bg = sharp(src, coef * 32);
	src_bg = sharp(src_bg, coef * 23);
	cvtColor(src_bg, src_bg, COLOR_BGR2GRAY);

	Mat background_lut = createLUT(colormap_background);	//creating background lut
	Mat halftone_lut = createLUT(colormap_halftone);		//creating lut for halftone effect

	return { src, dst, src_grayscale, src_bg,background_lut , halftone_lut, 100, 3 };	//start pack for PopArt function
};

int main(int argc, const char *argv[])
{

	if (argc < 2)
	{
		cerr << "We need an image to process here. Please run: popArt [path_to_image]" << endl;
		return -1;
	}

	Mat src = imread(argv[1], 1);

	if (src.empty())
	{
		cerr << "Image (" << argv[1] << ") is empty. Please adjust your path, so it points to a valid input image!" << endl;
		return -1;
	}


	//Creating mainwindow, trackbars, application PopArt effect for start effect amount, halftone size.
	ContextPopArt context_pop_art = setup(src);
	window_setup("Pop Art", src.rows, src.cols);
	PopArt(&context_pop_art);
	imshow("Pop Art", context_pop_art.dst);
	createTrackbar("Amount", "Pop Art", &context_pop_art.effect_amount, 100, updateAmount, &context_pop_art);
	createTrackbar("Halftone size", "Pop Art", &context_pop_art.halftone_radius, MAX_HALFTONE_SIZE, updatePopArt, &context_pop_art);

	//close program, clearing memory
	waitKey(0);
	destroyWindow("Pop Art");
	imwrite("result.png", context_pop_art.dst);
	context_pop_art.~ContextPopArt();

	return 0;
}
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <omp.h>

using namespace std;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void get_histogram(unsigned char* input, float* histogram, int width, int height, int step) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	const int index = yIndex * step + xIndex;

	if (xIndex < width && yIndex < height) {
		atomicAdd(&histogram[input[index]], 1);
	}
}

__global__ void normalize_histogram(float* histogram, float* histogram_normalized, int width, int height) {

	unsigned int nxy = threadIdx.x + threadIdx.y * blockDim.x;

	if (nxy < 256 && blockIdx.x == 0 && blockIdx.y == 0) {
		for (int i = 0; i < nxy; i++) {
			histogram_normalized[nxy] += histogram[i];
		}
		histogram_normalized[nxy] = histogram_normalized[nxy] * 255 / (width * height);
	}

}

__global__ void apply_histogram_image(unsigned char* input, unsigned char* output, float* histogram_normalized, int width, int height, int step) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	const int index = yIndex * step + xIndex;

	if (xIndex < width && yIndex < height) {
		output[index] = histogram_normalized[input[index]];
	}
}

void normalize_cpu(const cv::Mat& input, cv::Mat& output) {
	float histogram[256] = {};
	float histogram_normalized[256] = {};
	int i, j, k, l, m, n;
	int size = input.rows * input.cols;

	// Get number of processors
	int nProcessors = omp_get_max_threads();
	std::cout << "CPU processors available: " << nProcessors << std::endl;

	// Set number of processors to use with OpenMP
	omp_set_num_threads(6);

	#pragma omp parallel for private(i, j) shared(input, histogram)
	for (i = 0; i < input.rows; i++) {
		for (j = 0; j < input.cols; j++) {
			histogram[(int)input.at<uchar>(i, j)]++;
		}
	}

	#pragma omp parallel for private(k, l) shared(size, histogram, histogram_normalized)
	for (k = 0; k < 256; k++) {
		for (l = 0; l < k; l++) {
			histogram_normalized[k] += histogram[l];
		}
		histogram_normalized[k] = histogram_normalized[k] * 255 / size;
	}

	#pragma omp parallel for private(m, n) shared(input, output, histogram_normalized)
	for (m = 0; m < output.rows; m++) {
		for (n = 0; n < output.cols; n++) {
			output.at<uchar>(m, n) = histogram_normalized[(int)input.at<uchar>(m, n)];
		}
	}
}

void normalize_gpu(const cv::Mat& input, cv::Mat& output) {

	// Set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	SAFE_CALL(cudaSetDevice(dev), "Error setting device");

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t inputBytes = input.step * input.rows;
	size_t outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;
	float * d_histogram = {};
	float * d_histogram_normalized = {};

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_histogram, 256 * sizeof(float)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_histogram_normalized, 256 * sizeof(float)), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), outputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(32, 32);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((input.cols) / block.x, (input.rows) / block.y);

	// Launch the color conversion kernel
	auto start_cpu = chrono::high_resolution_clock::now();
	get_histogram << <grid, block >> > (d_input, d_histogram, input.cols, input.rows, static_cast<int>(input.step));
	normalize_histogram << <grid, block >> > (d_histogram, d_histogram_normalized, input.cols, input.rows);
	apply_histogram_image << <grid, block >> > (d_input, d_output, d_histogram_normalized, input.cols, input.rows, static_cast<int>(input.step));
	auto end_cpu = chrono::high_resolution_clock::now();

	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("GPU elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[])
{
	string imagePath;

	if (argc < 2)
		imagePath = "Images/woman3.jpg";
	else
		imagePath = argv[1];

	cout << imagePath << endl;
	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat grayscale_input;
	cvtColor(input, grayscale_input, cv::COLOR_BGR2GRAY);

	//creating output image
	cv::Mat output_gpu(grayscale_input.rows, grayscale_input.cols, grayscale_input.type());
	cv::Mat output_cpu(grayscale_input.rows, grayscale_input.cols, grayscale_input.type());

	// GPU
	normalize_gpu(grayscale_input, output_gpu);

	// CPU
	auto start_cpu = chrono::high_resolution_clock::now();
	normalize_cpu(grayscale_input, output_cpu);
	auto end_cpu = chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("CPU elapsed %f ms\n", duration_ms.count());

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output_GPU", cv::WINDOW_NORMAL);
	namedWindow("Output_CPU", cv::WINDOW_NORMAL);

	// output = input_bw.clone();
	imshow("Input", grayscale_input);
	imshow("Output_GPU", output_gpu);
	imshow("Output_CPU", output_cpu);

	//Wait for key press
	cv::waitKey();

	return 0;
}

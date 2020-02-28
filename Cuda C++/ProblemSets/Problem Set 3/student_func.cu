/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

//__global__ void findMaxMin(int numRows, int numCols,  float* max1,  float* min1,
//	 float* max2,  float* min2, int step) {
//	int X = threadIdx.x + blockIdx.x * blockDim.x;
//	int Y = threadIdx.y + blockIdx.y * blockDim.y;
//    if (X >= numCols || Y >= numRows || (Y * numCols + X + (1 << step)) > (numCols* numRows))
//		return;
//	int offset = Y * numCols + X;
//    if (step % 2 == 1) {
//        max2[offset] = MAX(max1[offset], max1[offset + (1 << step)]);
//        min2[offset] = MIN(min1[offset], min1[offset + (1 << step)]);
//    }
//    else {
//        max1[offset] = MAX(max2[offset], max2[offset + (1 << step)]);
//        min1[offset] = MIN(min2[offset], min2[offset + (1 << step)]);
//    }
//}

//__global__ void generateBins(int numRows, int numCols, const float* const logLuminance, 
//    unsigned int* histo, float logLumMin, float logLumRange, int numBins) {
//    int X = threadIdx.x + blockIdx.x * blockDim.x;
//    int Y = threadIdx.y + blockIdx.y * blockDim.y;
//    if (X >= numCols || Y >= numRows)
//        return;
//    int offset = Y * numCols + X;
//    unsigned int bin = MIN(static_cast<unsigned int>(numBins - 1),
//        static_cast<unsigned int>((logLuminance[offset] - logLumMin) / logLumRange * numBins));
//    //atomicAdd(histo[bin], 1);
//}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
	unsigned int* const d_cdf, float& min_logLum, float& max_logLum,
	const size_t numRows, const size_t numCols, const size_t numBins)
{
	//dim3 blockSize(32, 32, 1);
	//dim3 gridSize(((unsigned int)(numCols - 1) / 32 + 1), (unsigned int)((numCols - 1) / 32 + 1), 1);

	////TODO
	///*Here are the steps you need to implement */
	///*1) find the minimum and maximum value in the input logLuminance channel
	//store in min_logLum and max_logLum */
	//int imgSize = numCols * numCols;
	//float* d_max1, * d_min1, * d_max2, * d_min2;
	//checkCudaErrors(cudaMalloc(&d_max1, imgSize * sizeof(float)));
	//checkCudaErrors(cudaMalloc(&d_min1, imgSize * sizeof(float)));
	//checkCudaErrors(cudaMemcpy(&d_max1, d_logLuminance, imgSize * sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(&d_min1, d_logLuminance, imgSize * sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMalloc(&d_max2, imgSize * sizeof(float)));
	//checkCudaErrors(cudaMalloc(&d_min2, imgSize * sizeof(float)));
 //   int i = 1, step = 1;
 //   for (; (i << step) < imgSize; ++step)
 //       findMaxMin << <blockSize, gridSize >> > (numRows, numCols, d_min1, d_max1, d_max2, d_min2, step);
 //   if (step % 2) {
 //       checkCudaErrors(cudaMemcpy(&min_logLum, &d_min2, sizeof(float), cudaMemcpyDeviceToHost));
 //       checkCudaErrors(cudaMemcpy(&max_logLum, &d_max2, sizeof(float), cudaMemcpyDeviceToHost));
 //   }
 //   else {
 //       checkCudaErrors(cudaMemcpy(&min_logLum, &d_min1, sizeof(float), cudaMemcpyDeviceToHost));
 //       checkCudaErrors(cudaMemcpy(&max_logLum, &d_max1, sizeof(float), cudaMemcpyDeviceToHost));
 //   }

	///*2) subtract them to find the range*/
 //   float logLumRange = max_logLum - min_logLum;

	/*3) generate a histogram of all the values in the logLuminance channel using
	the formula: bin = (lum[i] - lumMin) / lumRange * numBins*/
    /*unsigned int* d_histo;
    checkCudaErrors(cudaMalloc(&d_histo, numBins * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(unsigned int)));*/
    /*generateBins << <blockSize, gridSize >> > (numRows, numCols, d_logLuminance, d_histo, min_logLum,
        logLumRange, numBins);*/

	/*4) Perform an exclusive scan (prefix sum) on the histogram to get
	the cumulative distribution of luminance values (this should go in the
	incoming d_cdf pointer which already has been allocated for you)*/


}

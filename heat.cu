#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants
const double DiffusionConstant = 0.25;    // Diffusion constant controlling the rate of heat transfer
const int ImageWidth = 200;               // Width of the simulated image
const int ImageHeight = 200;              // Height of the simulated image
const int BlockSize = 512;                // Number of threads per block in GPU kernel
const int NumIterations = 5000;           // Number of simulation iterations

using namespace std;

// GPU kernel to update temperature values based on the diffusion equation
__global__ void UpdateTemperature(double* next_temp, double* current_temp, int width, int height)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = width * height;

    if (index < total_elements)
    {
        // Update temperature values using the diffusion equation
        next_temp[index] = (1 - 4 * DiffusionConstant) * current_temp[index];
        // Set x and y values for current 
        int x = index % width;
        int y = index / width;

        // Update temperature based on neighboring cells using an approximation of the laplacian using a central difference
        if (x >= 1 && y >= 1) next_temp[index] += DiffusionConstant * current_temp[index - width - 1];
        if (x <= width - 2 && y >= 1) next_temp[index] += DiffusionConstant * current_temp[index - width + 1];
        if (x >= 1 && y <= height - 2) next_temp[index] += DiffusionConstant * current_temp[index + width - 1];
        if (x <= width - 2 && y <= height - 2) next_temp[index] += DiffusionConstant * current_temp[index + width + 1];

        index += blockDim.x * gridDim.x;
    }
}

// GPU kernel to convert temperature values to color
__global__ void TemperatureToColor(unsigned char* color_array, const double* temp_array, int width, int height) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < width * height) {
        // Normalize temperature values and map to RGB color
        double temperature = temp_array[index];
        double normalizedTemp = (temperature - 30000) / (1000 - 30000);
        normalizedTemp = fmax(0.0, fmin(normalizedTemp, 1.0));

        unsigned char r = 255;
        unsigned char g = static_cast<unsigned char>(255 * normalizedTemp);
        unsigned char b = static_cast<unsigned char>(255 * normalizedTemp);

        // Set RGB values in the color array
        color_array[index * 3] = r;
        color_array[index * 3 + 1] = g;
        color_array[index * 3 + 2] = b;
    }
}

int main() 
{
    // Host and device arrays for temperature and color data
    unsigned char* color_array = new unsigned char[ImageWidth * ImageHeight * 3];
    double* temp_array = new double[ImageWidth * ImageHeight];
    double* device_temp_array_current;
    double* device_temp_array_next;
    unsigned char* device_color_array;

    // Allocate device memory for temperature and color arrays
    cudaMalloc(&device_color_array, sizeof(unsigned char) * ImageWidth * ImageHeight * 3);
    cudaMalloc(&device_temp_array_current, sizeof(double) * ImageWidth * ImageHeight);
    cudaMalloc(&device_temp_array_next, sizeof(double) * ImageWidth * ImageHeight);

    // Initialize temperature array with initial conditions
    fill_n(temp_array, ImageWidth * ImageHeight, 1000);
    for (int y = 80; y < 120; ++y)
        for (int x = 80; x < 120; ++x)
            temp_array[y * ImageWidth + x] = 40000;

    // Copy initial temperature array to the device
    cudaMemcpy(device_temp_array_current, temp_array, sizeof(double) * ImageWidth * ImageHeight, cudaMemcpyHostToDevice);

    // Simulation loop
    for (int i = 0; i < NumIterations; ++i) 
    {
        // Update temperature values on the GPU
        UpdateTemperature<<<(ImageWidth * ImageHeight + BlockSize - 1) / BlockSize, BlockSize>>>(
            i % 2 == 0 ? device_temp_array_next : device_temp_array_current,
            i % 2 == 0 ? device_temp_array_current : device_temp_array_next, ImageWidth, ImageHeight);

        // Convert temperature values to color on the GPU
        TemperatureToColor<<<(ImageWidth * ImageHeight + BlockSize - 1) / BlockSize, BlockSize>>>(
            device_color_array, i % 2 == 0 ? device_temp_array_next : device_temp_array_current, ImageWidth, ImageHeight);

        // Copy color data from the device to the host
        cudaMemcpy(color_array, device_color_array, sizeof(unsigned char) * ImageWidth * ImageHeight * 3, cudaMemcpyDeviceToHost);
        
        // Save the frame as a PNG image on the host
        char filename[128];
        sprintf(filename, "frame_%04d.png", i);
        stbi_write_png(filename, ImageWidth, ImageHeight, 3, color_array, ImageWidth * 3);
    }

    // Clean up allocated memory
    delete[] color_array;
    delete[] temp_array;
    cudaFree(device_color_array);
    cudaFree(device_temp_array_current);
    cudaFree(device_temp_array_next);

    return 0;
}

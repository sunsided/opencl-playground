// following https://anteru.net/blog/2012/11/04/2016/index.html
// TODO:     http://stackoverflow.com/a/28644721/195651

#include "CL/cl.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <ctime>

using namespace std;

void printDeviceInformation(const vector<cl_platform_id> platformIds);
string loadKernelCodeFromFile(const char* name);
cl_program createProgram(const string& source, cl_context context);
void printProgramBuildErrorIfAny(vector<cl_device_id> deviceIds, cl_int error, cl_program program);
void executeSaxpy(cl_uint deviceIdCount, vector<cl_device_id> deviceIds, cl_context context);
void executeConvolution(cv::Mat src, cl_uint deviceIdCount, vector<cl_device_id> deviceIds, cl_int& error, cl_context context);

const string window_name = "Test Window";

void checkError(cl_int error)
{
    if (error != CL_SUCCESS) {
        cerr << "OpenCL call failed with error " << error << endl << flush;
        exit(1);
    }
}

cv::Mat loadImageBGRA(const string &path)
{
    cout << "Loading image ..." << endl;
    auto original = cv::imread(path, cv::IMREAD_COLOR);
    assert(original.dims > 0);

    cv::Mat originalWithAlpha(original.size(), CV_MAKE_TYPE(original.type(), 4));

    int from_to[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
    cv::mixChannels(&original, 2, &originalWithAlpha, 1, from_to, 4);

    return originalWithAlpha;
}

void main()
{
    cout << "Loading image ..." << endl;
    auto src = loadImageBGRA("../images/emily_browning.jpg");
    assert(src.dims > 0);

#if 0
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(window_name, src);
    cv::waitKey(0);
    return;
#endif

    // get the number of platforms
    cl_uint platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &platformIdCount);

    // get all platforms
    vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

    printDeviceInformation(platformIds);

    // get the number of devices on the first platform
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

    // get the devices on the first platform
    vector<cl_device_id> deviceIds(deviceIdCount);
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

    // get a context on the first platform
    const cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties> (platformIds[0]),
        0, 0
    };

    cout << "Creating the context ..." << endl;
    cl_int error;
    auto context = clCreateContext(
        contextProperties, deviceIdCount,
        deviceIds.data(), nullptr,
        nullptr, &error);

    // run the SAXPY test
    cout << "Running SAXPY ..." << endl;
    executeSaxpy(deviceIdCount, deviceIds, context);

    // run convolution test
    cout << "Running convolution ..." << endl;  
    executeConvolution(src, deviceIdCount, deviceIds, error, context);


    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(window_name, src);
    cv::waitKey(0);



    cout << "Releasing the context ..." << endl;
    clReleaseContext(context);
}

string loadKernelCodeFromFile(const char* filename)
{
    ifstream in(filename);
    string result((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    return result;
}

cl_program createProgram(const string& source, cl_context context)
{
    size_t lengths[1] = { source.size() };
    const char* sources[1] = { source.data() };

    cl_int error = 0;
    auto program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
    checkError(error);

    return program;
}

void printDeviceInformation(const vector<cl_platform_id> platformIds)
{
    for (const auto platformId : platformIds)
    {
        size_t length;

        // get the platforn name
        clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 0, nullptr, &length);
        auto* platformName = new char[length];
        clGetPlatformInfo(platformId, CL_PLATFORM_NAME, length, platformName, nullptr);

        // get the platform version
        clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 0, nullptr, &length);
        auto* platformVersion = new char[length];
        clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, length, platformVersion, nullptr);

        cout << platformName << " (" << platformVersion << ")" << endl;
        delete[] platformName;
        delete[] platformVersion;

        // get the number of devices on the platform
        cl_uint deviceIdCount = 0;
        clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

        // get the devices on the platform
        vector<cl_device_id> deviceIds(deviceIdCount);
        clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

        for (const auto deviceId : deviceIds)
        {
            // get the device name
            clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &length);
            auto* deviceName = new char[length];
            clGetDeviceInfo(deviceId, CL_DEVICE_NAME, length, deviceName, nullptr);

            cout << "  - " << deviceName << endl;
            delete[] deviceName;
        }
    }
}

void executeSaxpy(cl_uint deviceIdCount, vector<cl_device_id> deviceIds, cl_context context)
{
    cout << "Creating the program ..." << endl;
    const auto kernelCode = loadKernelCodeFromFile("saxpy.cl");
    auto program = createProgram(kernelCode, context);

    cout << "Building the program ..." << endl;
    auto error = clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);
    printProgramBuildErrorIfAny(deviceIds, error, program);
    checkError(error);

    cout << "Creating the kernel ..." << endl;
    auto kernel = clCreateKernel(program, "SAXPY", &error);     // SAXPY is Y = a*X + Y
    checkError(error);

    // prepare some test data
    static const size_t testDataSize = 1 << 10;
    vector<float> a(testDataSize), b(testDataSize);
    for (auto i = 0; i < testDataSize; ++i) {
        a[i] = static_cast<float> (i);                          // "X" in SAXPY
        b[i] = static_cast<float> (10);                         // "Y" in SAXPY, so Y = a*i+10
    }

    // buffer for the first parameter
    auto aBuffer = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,                // read-only for the kernel, automatically upload from the host
        sizeof(float) * (testDataSize),
        a.data(), &error);
    checkError(error);

    // buffer for the second parameter
    auto bBuffer = clCreateBuffer(context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,               // read/write for the kernel, automatically upload from the host
        sizeof(float) * (testDataSize),
        b.data(), &error);
    checkError(error);

    // the command queue
    cout << "Creating the command queue ..." << endl;
    auto queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);
    checkError(error);

    // set the parameters for the kernel (positional)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    static const float two = 2.0f;
    clSetKernelArg(kernel, 2, sizeof(float), &two);

    // enqueue the command to execute the kernel
    cout << "Enqueueing the kernel execution ..." << endl;
    const size_t globalWorkSize[] = { testDataSize, 0, 0 };
    error = clEnqueueNDRangeKernel(queue, kernel,
        1,                                  // work_dim
        nullptr,                            // global_work_offset
        globalWorkSize,
        nullptr,                            // local_work_size
        0, nullptr, nullptr);
    checkError(error);

    // enqueue the command to read the results back
    cout << "Enqueueing the blocking memory read ..." << endl;
    error = clEnqueueReadBuffer(queue, bBuffer,
        CL_TRUE,                            // blocking_read
        0,                                  // offset
        sizeof(float) * testDataSize,       // number of bytes to read
        b.data(),                           // target pointer in host memory
        0, nullptr, nullptr);
    checkError(error);

    cout << "Releasing the command queue ..." << endl;
    clReleaseCommandQueue(queue);

    cout << "Releasing the buffers ..." << endl;
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(aBuffer);

    cout << "Releasing the kernel ..." << endl;
    clReleaseKernel(kernel);

    cout << "Releasing the program ..." << endl;
    clReleaseProgram(program);
}

void executeConvolution(cv::Mat src, cl_uint deviceIdCount, vector<cl_device_id> deviceIds, cl_int& error, cl_context context)
{
    // generate a filter kernel
    // http://dev.theomader.com/gaussian-kernel-calculator/
    // TODO: split the kernel, perform two operations
    const auto filterRadius = 11;
    float filter[] = {
        0.0071, 0.007427, 0.007691, 0.007886, 0.008005, 0.008045, 0.008005, 0.007886, 0.007691, 0.007427, 0.0071,
        0.007427, 0.007768, 0.008045, 0.008248, 0.008373, 0.008415, 0.008373, 0.008248, 0.008045, 0.007768, 0.007427,
        0.007691, 0.008045, 0.008331, 0.008542, 0.008671, 0.008714, 0.008671, 0.008542, 0.008331, 0.008045, 0.007691,
        0.007886, 0.008248, 0.008542, 0.008758, 0.00889, 0.008935, 0.00889, 0.008758, 0.008542, 0.008248, 0.007886,
        0.008005, 0.008373, 0.008671, 0.00889, 0.009025, 0.00907, 0.009025, 0.00889, 0.008671, 0.008373, 0.008005,
        0.008045, 0.008415, 0.008714, 0.008935, 0.00907, 0.009115, 0.00907, 0.008935, 0.008714, 0.008415, 0.008045,
        0.008005, 0.008373, 0.008671, 0.00889, 0.009025, 0.00907, 0.009025, 0.00889, 0.008671, 0.008373, 0.008005,
        0.007886, 0.008248, 0.008542, 0.008758, 0.00889, 0.008935, 0.00889, 0.008758, 0.008542, 0.008248, 0.007886,
        0.007691, 0.008045, 0.008331, 0.008542, 0.008671, 0.008714, 0.008671, 0.008542, 0.008331, 0.008045, 0.007691,
        0.007427, 0.007768, 0.008045, 0.008248, 0.008373, 0.008415, 0.008373, 0.008248, 0.008045, 0.007768, 0.007427,
        0.0071, 0.007427, 0.007691, 0.007886, 0.008005, 0.008045, 0.008005, 0.007886, 0.007691, 0.007427, 0.0071
    };
    assert((sizeof(filter) / sizeof(filter[0])) == (filterRadius*filterRadius));

    auto define = string("-D FILTER_SIZE=") + to_string(filterRadius);

    cout << "Creating the program ..." << endl;
    const auto kernelCode = loadKernelCodeFromFile("convolution.cl");
    auto program = createProgram(kernelCode, context);

    cout << "Building the program ..." << endl;
    error = clBuildProgram(program, deviceIdCount, deviceIds.data(), define.c_str(), nullptr, nullptr);
    printProgramBuildErrorIfAny(deviceIds, error, program);
    checkError(error);

    cout << "Creating the kernel ..." << endl;
    auto kernel = clCreateKernel(program, "Convolution", &error);
    checkError(error);

    cout << "Creating the input image ..." << endl;
    static const cl_image_format format = { CL_BGRA, CL_UNORM_INT8 };
    auto inputImage = clCreateImage2D(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
        src.cols, src.rows, 0,
        src.ptr(),
        &error);
    checkError(error);

    // removing all traces of the original image; magenta for extra ugliness
    src.setTo(cv::Scalar(255, 0, 255));

    cout << "Creating the output image ..." << endl;
    auto outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, src.cols, src.rows, 0, nullptr, &error);
    checkError(error);

    cout << "Creating buffer for filter weights ..." << endl;
    auto filterWeightsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(filter), filter, &error);
    checkError(error);

        cout << "Creating the command queue ..." << endl;
    auto queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);
    checkError(error);

    // timings ftw.
    auto startTime = clock();

    // wire up the kernel parameters
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterWeightsBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputImage);
    
    cout << "Enqueueing the kernel operation ..." << endl;
    std::size_t offset[3] = { 0 };
    std::size_t size[3] = { src.cols, src.rows, 1 };
    error = clEnqueueNDRangeKernel(queue, kernel, 2, offset, size, nullptr, 0, nullptr, nullptr);
    checkError(error);
    
    cout << "Enqueueing the read operation ..." << endl;
    std::size_t origin[3] = { 0 };
    std::size_t region[3] = { src.cols, src.rows, 1 };
    clEnqueueReadImage(queue, outputImage, CL_TRUE,
        origin, region, 0, 0,
        src.ptr(), 0, nullptr, nullptr);

    // timings ftw.
    auto duration = (double)(clock() - startTime) / CLOCKS_PER_SEC * 1000.0;;
    cout << "Total execution time: " << duration << "ms" << endl;

    cout << "Releasing the command queue ..." << endl;
    clReleaseCommandQueue(queue);

    cout << "Releasing the buffers ..." << endl;
    clReleaseMemObject(inputImage);
    clReleaseMemObject(outputImage);
    clReleaseMemObject(filterWeightsBuffer);

    cout << "Releasing the kernel ..." << endl;
    clReleaseKernel(kernel);

    cout << "Releasing the program ..." << endl;
    clReleaseProgram(program);
}

void printProgramBuildErrorIfAny(vector<cl_device_id> deviceIds, cl_int error, cl_program program)
{
    if (CL_BUILD_PROGRAM_FAILURE == error) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        // Allocate memory for the log
        auto log = new char[log_size];

        // Get the log
        clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

        // Print the log
        cerr << "Kernel compilation failed: " << log << endl;
        delete[] log;
    }
}
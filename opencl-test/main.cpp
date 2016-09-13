// following https://anteru.net/blog/2012/11/04/2016/index.html

#include "CL/cl.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;

void printDeviceInformation(const vector<cl_platform_id> platformIds);
string loadKernelCodeFromFile(const char* name);
cl_program createProgram(const string& source, cl_context context);

void checkError(cl_int error)
{
    if (error != CL_SUCCESS) {
        cerr << "OpenCL call failed with error " << error << endl << flush;
        exit(1);
    }
}

void main()
{
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

    cout << "Creating the program ..." << endl;
    const auto kernelCode = loadKernelCodeFromFile("saxpy.cl");
    auto program = createProgram(kernelCode, context);

    cout << "Building the program ..." << endl;
    error = clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);
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

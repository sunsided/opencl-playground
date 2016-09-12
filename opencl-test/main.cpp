// following https://anteru.net/blog/2012/11/04/2016/index.html

#include "CL/cl.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;

void checkError(cl_int error)
{
    if (error != CL_SUCCESS) {
        cerr << "OpenCL call failed with error " << error << endl << flush;
        exit(1);
    }
}

string loadKernel(const char* name)
{
    ifstream in(name);
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

void main()
{
    // get the number of platforms
    cl_uint platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &platformIdCount);

    // get all platforms
    vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

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

        cout << platformName << " (" << platformVersion << ")" <<  endl;
        delete[] platformName;
        delete[] platformVersion;

        // get the number of defines on the first platform
        cl_uint deviceIdCount = 0;
        clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

        // get the devices on the first platform
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

    // get the number of defines on the first platform
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

    // get the devices on the first platform
    vector<cl_device_id> deviceIds(deviceIdCount);
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

    // get a context on the first platform
    const cl_context_properties contextProperties[] =
    {
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
    const auto kernelCode = loadKernel("saxpy.cl");
    auto program = createProgram(kernelCode, context);

    cout << "Building the program ..." << endl;
    error = clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);
    checkError(error);

    cout << "Creating the kernel ..." << endl;
    auto kernel = clCreateKernel(program, "SAXPY", &error);
    checkError(error);

    cout << "Releasing the kernel ..." << endl;
    clReleaseKernel(kernel);

    cout << "Releasing the program ..." << endl;
    clReleaseProgram(program);

    cout << "Releasing the context ..." << endl;
    clReleaseContext(context);
}

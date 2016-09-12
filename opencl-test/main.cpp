#include "CL/cl.h"
#include <vector>
#include <iostream>
#include <cstdint>

using namespace std;

void main()
{
    // get the number of platforms
    cl_uint platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &platformIdCount);

    // get all platforms
    std::vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

    for (const auto id : platformIds)
    {
        size_t length;
        clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &length);

        auto* value = new char[length];
        clGetPlatformInfo(id, CL_PLATFORM_NAME, length, value, nullptr);

        cout << "Platform " << id << ": " << value << endl;
        delete[] value;
    }

    // get the number of defines on the first platform
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

    // get the devices on the first platform
    std::vector<cl_device_id> deviceIds(deviceIdCount);
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);
}

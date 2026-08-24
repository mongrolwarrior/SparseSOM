#pragma once
namespace sparsesom {
float       kernel_self_test();    // runs a trivial kernel; returns 5.0f if the GPU path works
const char* device_name();
}
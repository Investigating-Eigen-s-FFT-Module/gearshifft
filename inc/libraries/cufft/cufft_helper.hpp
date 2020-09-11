#ifndef CUFFT_HELPER_HPP_
#define CUFFT_HELPER_HPP_

#include "core/get_memory_size.hpp"
#include "core/unused.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <cuda_runtime.h>
#include <cufft.h>
#pragma GCC diagnostic pop

#include <sstream>
#include <stdexcept>

#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) gearshifft::CuFFT::check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) gearshifft::CuFFT::check_cuda(cudaGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_LAST(msg) {}
#endif

namespace gearshifft {
namespace CuFFT {

  inline
  void throw_error(int code,
                   const char* error_string,
                   const char* msg,
                   const char* func,
                   const char* file,
                   int line) {
    throw std::runtime_error("CUDA error "
                             +std::string(msg)
                             +" "+std::string(error_string)
                             +" ["+std::to_string(code)+"]"
                             +" "+std::string(file)
                             +":"+std::to_string(line)
                             +" "+std::string(func)
      );
  }

  static const char* cufftResultToString(cufftResult error) {
    switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    }
    return "<unknown>";
  }

  inline
  void check_cuda(cudaError_t code, const char* msg, const char *func, const char *file, int line) {
    if (code != cudaSuccess) {
      throw_error(static_cast<int>(code),
                  cudaGetErrorString(code), msg, func, file, line);
    }
  }
  inline
  void check_cuda(cufftResult code, const char* msg,  const char *func, const char *file, int line) {
    gearshifft::ignore_unused(msg);

    if (code != CUFFT_SUCCESS) {
      throw_error(static_cast<int>(code),
                  cufftResultToString(code), "cufft", func, file, line);
    }
  }

  inline
  std::stringstream getCUDADeviceInformations(int dev) {
    std::stringstream info;
    cudaDeviceProp prop;
    int runtimeVersion = 0;
    int cufftv = 0;
    size_t f=0, t=0;
    CHECK_CUDA( cufftGetVersion(&cufftv) );
    CHECK_CUDA( cudaRuntimeGetVersion(&runtimeVersion) );
    CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
    CHECK_CUDA( cudaMemGetInfo(&f, &t) );
    info << '"' << prop.name << '"'
         << ", \"CC\", " << prop.major << '.' << prop.minor
         << ", \"PCI Bus ID\", " << prop.pciBusID
         << ", \"PCI Device ID\", " << prop.pciDeviceID
         << ", \"Multiprocessors\", "<< prop.multiProcessorCount
         << ", \"Memory [MiB]\", "<< t/1048576
         << ", \"MemoryFree [MiB]\", " << f/1048576
         << ", \"HostMemory [MiB]\", "<< getMemorySize()/1048576
         << ", \"ECC enabled\", " << prop.ECCEnabled
         << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
         << ", \"GPUClock [MHz]\", " << prop.clockRate/1000
         << ", \"CUDA Runtime\", " << runtimeVersion
         << ", \"cufft\", " << cufftv
      ;
    return info;
  }

  std::stringstream listCudaDevices() {
    std::stringstream info;
    int nrdev = 0;
    CHECK_CUDA( cudaGetDeviceCount( &nrdev ) );
    if(nrdev==0)
      throw std::runtime_error("No CUDA capable device found");
    for(int i=0; i<nrdev; ++i)
      info << "\"ID\"," << i << "," << getCUDADeviceInformations(i).str() << std::endl;
    return info;
  }

  bool deviceSupportsHalfPrecision(int dev) {
    cudaDeviceProp prop;
    CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
    return ( (prop.major==5 && prop.minor>=3)
            || prop.major>=6 );
  }
} // CuFFT
} // gearshifft
#endif

#ifndef EIGEN_HPP_
#define EIGEN_HPP_

#include "core/types.hpp"
#include "core/options.hpp"
#include "core/context.hpp"
#include "core/application.hpp"
#include "core/timer.hpp"
#include "core/fft.hpp"
#include "core/benchmark_suite.hpp"
#include "core/get_memory_size.hpp"
#include "core/unused.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>

#define EIGEN_RUNTIME_NO_MALLOC                 // ensure that the allocate() allocates all memory and there
                                                // is no dynamic allocation during other stages
                                                // todo: remove once certain that no assertions are broken
                                                //       as it adds overhead in measurements
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include <fftw3.h>
#include <eigen3/unsupported/Eigen/src/FFT/ei_fftw_impl.h> // include ei_fftw_impl
                                                           // directly as opposed to defining
                                                           // EIGEN_FFTW_DEFAULT
                                                           // This should make switching
                                                           // backends easier

namespace gearshifft
{
  namespace eigen
  {
    class EigenOptions : public OptionsDefault
    {
      public:
        EigenOptions() : OptionsDefault() {

        }; // todo: add different back-ends as options
    };

    namespace traits
    {
      template <typename T_Precision = float>
      struct plan
      {
        using ComplexType = std::complex<T_Precision>;
        using RealType = T_Precision;
      };

      template <>
      struct plan<double>
      {
        using ComplexType = std::complex<double>;
        using RealType = double;
      };
    }

    struct EigenContext : public ContextDefault<EigenOptions>
    {
      static const std::string title()
      {
        // todo: differentiate different back-ends
        return "eigen";
      }

      static std::string get_device_list()
      {
        // todo: update for multithread support
        std::ostringstream msg;

        msg << "Only single thread CPU supported.\n";

        return msg.str();
      }

      std::string get_used_device_properties()
      {
        std::ostringstream msg;

        msg << "\"SupportedThreads\"," << 1
            << ",\"UsedThreads\"," << 1
            << ",\"TotalMemory\"," << getMemorySize();

        return msg.str();
      }
    };

    template <typename TFFT,       // see fft.hpp (FFT_Inplace_Real, ...)
              typename TPrecision, // double
              size_t NDim          // 1 todo: allow 2 and 3 with for loop
              >
    struct EigenImpl
    {
      //////////////////////////////////////////////////////////////////////////////////////
      // COMPILE TIME FIELDS

      using Extent = std::array<std::size_t, NDim>;
      // todo: Eigen with FFTW backend only does ESTIMATE as of now!
      using ComplexType = typename traits::plan<TPrecision>::ComplexType;
      using RealType = typename traits::plan<TPrecision>::RealType;
      #if NDim != 1
      // todo: what's a solution for this...
      // #pragma message("WARNING: gearshifft_eigen only does 1D and will treat all dimensions with NDim == 1 for now\n")
      #endif

      static constexpr bool IsComplex = TFFT::IsComplex;

      // todo: missing options: FFT(const impl_type& impl = impl_type(), Flag flags = Default)
      using fft_wrapper_type = typename Eigen::FFT<TPrecision, Eigen::internal::fftw_impl<TPrecision>>;
      using value_type = typename std::conditional<IsComplex, ComplexType, RealType>::type;
      using data_type = Eigen::Matrix<
        value_type,
        Eigen::Dynamic,
        1 /*NDim*/,
        Eigen::ColMajor
      >;
      using data_complex_type = Eigen::Matrix<
        ComplexType,
        Eigen::Dynamic,
        1 /*NDim*/,
        Eigen::ColMajor
      >;

      //////////////////////////////////////////////////////////////////////////////////////

      /// extents of the FFT input data
      Extent extents_ = {{0}};

      /// product of corresponding extents
      size_t n_ = 0;

      data_type* data_ = nullptr;     
      data_complex_type* data_complex_ = nullptr;

      /// size in nr of elements(!) of FFT input data
      size_t data_size_ = 0;
      /// size in nr of elements(!) of FFT(input) for out-of-place transforms
      size_t data_complex_size_ = 0;

      fft_wrapper_type eigen_fft_;

      EigenImpl(const Extent &cextents)
      {
        Eigen::internal::set_is_malloc_allowed(false); // disable eigen internal malloc
        
        extents_ = interpret_as::column_major(cextents);

        // is nfft in eigen implementation
        n_ = std::accumulate(extents_.begin(),
                             extents_.end(),
                             1,
                             std::multiplies<std::size_t>());

        data_size_ = n_;
        data_complex_size_ = data_size_;

        size_t total_mem = 95 * getMemorySize() / 100; // keep some memory available, otherwise an out-of-memory killer becomes more likely
        if (total_mem < 3 * data_size_ * sizeof(value_type) + data_complex_size_ * sizeof(ComplexType))
        { // includes host input buffers
          std::stringstream ss;
          ss << total_mem << "<" << 3 * data_size_ * sizeof(value_type) + data_complex_size_ * sizeof(ComplexType) << " (bytes)";
          throw std::runtime_error("FFT data exceeds physical memory. " + ss.str());
        }
      }

      ~EigenImpl()
      {
        destroy();
      }

      void allocate()
      {
        Eigen::internal::set_is_malloc_allowed(true); // enable eigen internal malloc
        data_ = new data_type(data_size_);
        data_complex_ = new data_complex_type(data_complex_size_);
      }

      void destroy()
      {
        if(data_)
          delete data_;
        data_ = nullptr;

        if(data_complex_)
          delete data_complex_;
        data_complex_ = nullptr;
      }

      /**
       * Returns allocated memory for FFT
       */
      size_t get_allocation_size()
      {
        return data_size_ * sizeof(value_type) + data_complex_size_ * sizeof(ComplexType);
      }

      /**
       * Returns size in bytes of one data transfer.
       *
       * Upload and download have the same size due to round-trip FFT.
       * \return Size in bytes of FFT data to be transferred (to device or to host memory buffer).
       */
      size_t get_transfer_size() {
        return data_size_ * sizeof(value_type);
      }

      // ignoring for now
      size_t get_plan_size()
      {
        return 0;
      }

      // create FFT plan handles
      // Currently Eigen has plan creation tightly linked to an immediate dft execute.
      // Without source code changes it's likely impossible to benchmark plan creation by itself
      // todo: maybe with eigen_fft_.impl() you can get to it...
      void init_forward()
      {
        Eigen::internal::set_is_malloc_allowed(true); // enable eigen internal malloc
        // re-call constructor (can also add opts later here)
        eigen_fft_ = fft_wrapper_type(Eigen::internal::fftw_impl<TPrecision>(),
                                      fft_wrapper_type::HalfSpectrum);
        // Plan creation will happen in warmup rounds hopefully, todo: I think I can make this better
        // with some ugly tinkering so that plan creation actually happens here.
      }
      void init_inverse()
      {
        Eigen::internal::set_is_malloc_allowed(true); // enable eigen internal malloc
        // Plan creation will happen in warmup rounds hopefully
      }

      void execute_forward()
      {
        Eigen::internal::set_is_malloc_allowed(true); // disable eigen internal malloc
        eigen_fft_.fwd(*data_complex_, *data_);
      }

      void execute_inverse()
      {
        Eigen::internal::set_is_malloc_allowed(true); // disable eigen internal malloc
        eigen_fft_.inv(*data_, *data_complex_);
      }

      // todo: how does reuseplan work? fft.hpp just seems to call init_inverse at
      // different points depending on whether it is set. I'm not sure how this works.

      // Assumes that THostData is a pointer to a value_type
      template <typename THostData>
      void upload(THostData *input)
      {
        static_assert(std::is_same<THostData,value_type>::value
                      && "upload(THostData *input) gave mismatched value type.");
        // Todo: this will probably break with more than 1D Matrices and/or outer stride
        //       once that is an option
        std::memcpy(data_->data(), input, data_size_ * sizeof(value_type));
      }

      // Assumes that THostData is a pointer to a value_type
      template <typename THostData>
      void download(THostData *output)
      {
        static_assert(std::is_same<THostData,value_type>::value
                      && "download(THostData *output) gave mismatched value type.");
        std::memcpy(output, data_->data(), data_size_ * sizeof(value_type));
      }
    };

    // InPlace not possible through Eigen API

    // Note: for uneven sizes, this will create mismatch as the Eigen API
    //       sets the output size for a complex to real ifft to (2 * (src.size()-1))
    //       as opposed to just reading dst.size(). It never assumes that dst.size()
    //       has been set, which is where this guesswork is from.
    //       todo: somehow skip radix357 runs  on this instantiation?
    using Outplace_Real = gearshifft::FFT<FFT_Outplace_Real,
                                          FFT_Plan_Not_Reusable,
                                          EigenImpl,
                                          TimerCPU>;

    using Outplace_Complex = gearshifft::FFT<FFT_Outplace_Complex,
                                             FFT_Plan_Not_Reusable,
                                             EigenImpl,
                                             TimerCPU>;
  } // namespace eigen
} // namespace gearshifft
#endif /* EIGEN_HPP_ */
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

// fft back-end selection at runtime proved impossible without Eigen API changes or overhead in benchmarks
// Instead, several targets will be created using the FFT header's macros.
// The title for the csv is set here.
#ifdef EIGEN_FFTW_DEFAULT
#define TITLE "eigen-fftw"
#elif defined EIGEN_MKL_DEFAULT
#define TITLE "eigen-mkl"
#include <mkl.h>
#elif defined EIGEN_POCKETFFT_DEFAULT
#define TITLE "eigen-pocketfft"
#else
#define TITLE "eigen-kissfft"
#endif

namespace gearshifft
{
  namespace eigen
  {
    class EigenOptions : public OptionsDefault
    {
      public:
        EigenOptions() : OptionsDefault() {
          add_options()
          ("scaling", value(&scaling_)->default_value("scaled"), "Normalize output (scaled/unscaled).")
          ("spectrum", value(&spectrum_)->default_value("full"), "Half or Full Spectrum on real fft/ifft (full/half).");
        };
        
        int flags() const {
          int flags = 0;
          if(scaling_ == "unscaled")
            flags |= 1;
          if(spectrum_ == "half")
            flags |= 2;
          return flags;
        }

        bool is_normalized() {
          return (scaling_ == "scaled");
        }
        
        std::string scaling() {
          return scaling_;
        }

        std::string spectrum() {
          return spectrum_;
        }

      private:
        std::string scaling_;
        std::string spectrum_;
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
        return TITLE;
      }

      static std::string get_device_list()
      {
        std::ostringstream msg;
        #ifdef EIGEN_MKL_DEFAULT
        msg << std::thread::hardware_concurrency() << " CPU threads supported.\n";
        #else
        msg << "Only single thread CPU supported.\n";
        #endif

        return msg.str();
      }

      std::string get_used_device_properties()
      {
        #ifdef EIGEN_MKL_DEFAULT
        size_t maxndevs = std::thread::hardware_concurrency();
        size_t ndevs = options().getNumberDevices();
        if(maxndevs == 0)
          maxndevs = 1;
        if(ndevs == 0 || ndevs > maxndevs)
          ndevs = maxndevs;
        #else
        size_t maxndevs = 1;
        size_t ndevs = 1;
        #endif
        std::ostringstream msg;
        msg << "\"SupportedThreads\"," << maxndevs
            << ",\"UsedThreads\"," << ndevs
            << ",\"TotalMemory\"," << getMemorySize()
            << ",\"Scaling\"," << options().scaling()
            << ",\"Spectrum\"," << options().spectrum();

        return msg.str();
      }
    };

    template <typename TFFT,       // see fft.hpp (FFT_Inplace_Real, ...)
              typename TPrecision, // foat, double
              size_t NDim          // overriden to 1, 2 and 3D not allowed (todo: some way to skip those?)
              >
    struct EigenImpl
    {
      //////////////////////////////////////////////////////////////////////////////////////
      // COMPILE TIME FIELDS

      using Extent = std::array<std::size_t, NDim>;
      using ComplexType = typename traits::plan<TPrecision>::ComplexType;
      using RealType = typename traits::plan<TPrecision>::RealType;
      #if NDim != 1
      // todo: what's a solution for this...
      // #pragma message("WARNING: gearshifft_eigen only does 1D and will treat all dimensions with NDim == 1 for now\n")
      #endif

      static constexpr bool IsComplex = TFFT::IsComplex;

      using fft_wrapper_type = typename Eigen::FFT<TPrecision>;
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
      int eigen_fft_flags_ = EigenContext::options().flags();

      EigenImpl(const Extent &cextents)
      {
        #ifdef EIGEN_MKL_DEFAULT
        // NOTE: according to doc, the number serves as a hint and MKL may opt to use less!
        mkl_set_num_threads(EigenContext::options().getNumberDevices());
        #endif
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
        // re-call constructor
        eigen_fft_ = fft_wrapper_type(Eigen::default_fft_impl<TPrecision>(),
                                      eigen_fft_flags_);
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

    // Inplace not possible with half spectrum enabled; since that flag is set
    // at run-time, I'll omit it for now. (todo): maybe add inplace complex later.
    // using Eigen::Map and specific parameters on some back-ends.
    // Note: for uneven sizes, when half spectrum is enabled,
    // this will create mismatch as the Eigen API sets the output size
    // for a complex to real ifft to (2 * (src.size()-1))
    // as opposed to just reading dst.size(). It never assumes that dst.size()
    // has been set, which is where this guesswork is from.
    // (todo): somehow skip those runs on this instantiation or at least add a warning?
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
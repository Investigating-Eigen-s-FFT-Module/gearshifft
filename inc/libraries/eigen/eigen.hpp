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

#include <eigen3/Eigen/Dense>
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

      struct memory_api {

        template <typename S, typename D>
        static D *memcpy(D *destination, const S *source, size_t size) {

          // If source type is_array, i.e. std::complex, reinterpret destination as array, too;
          // otherwise keep the original type.
          using R = std::conditional_t<std::is_array<S>::value, S, D>;

          return static_cast<D *>(std::memcpy(reinterpret_cast<R *>(destination), source, size));
        }
        
        template <typename S>
        static S* malloc(size_t size) {
          return static_cast<S*>(std::malloc(size * sizeof(S)));
        }

        static void free(void* p) { std::free(p); }
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
      // Eigen with FFTW backend only does ESTIMATE as of now!
      using ComplexType = typename traits::plan<TPrecision>::ComplexType;
      using RealType = typename traits::plan<TPrecision>::RealType;
      using MemoryAPI = typename traits::memory_api;
      #if NDim != 1
      // #pragma message("WARNING: gearshifft_eigen only does 1D and will treat all dimensions with NDim == 1 for now\n")
      #endif

      static constexpr bool IsInplace = TFFT::IsInplace;
      static constexpr bool IsComplex = TFFT::IsComplex;
      static constexpr bool IsInplaceReal = IsInplace && !IsComplex;

      // todo: missing options: FFT(const impl_type& impl = impl_type(), Flag flags = Default)
      using fft_wrapper_type = typename Eigen::FFT<TPrecision, Eigen::internal::fftw_impl<TPrecision>>;
      using value_type = typename std::conditional<IsComplex, ComplexType, RealType>::type;
      using eigen_data_type = Eigen::Matrix<
        value_type,
        Eigen::Dynamic,
        1 /*NDim*/,
        Eigen::ColMajor
      >;
      using eigen_data_complex_type = Eigen::Matrix<
        ComplexType,
        Eigen::Dynamic,
        1 /*NDim*/,
        Eigen::ColMajor
      >;
      using eigen_map_data_type = typename Eigen::Map<eigen_data_type>;

      using eigen_map_data_complex_type = typename Eigen::Map<eigen_data_complex_type>;


      //////////////////////////////////////////////////////////////////////////////////////

      /// extents of the FFT input data
      Extent extents_ = {{0}};
      /// extents of the FFT complex data (=FFT(input))
      Extent extents_complex_ = {{0}};
      /// product of corresponding extents
      size_t n_ = 0;
      /// product of corresponding extents
      size_t n_complex_ = 0;

      value_type* data_ = nullptr;     
      ComplexType* data_complex_ = nullptr;
      eigen_map_data_type* eigen_data_ = nullptr;
      eigen_map_data_complex_type* eigen_data_complex_ = nullptr;

      /// size in nr of elements(!) of FFT input data
      size_t data_size_ = 0;
      /// size in nr of elements(!) of FFT(input) for out-of-place transforms
      size_t data_complex_size_ = 0;

      fft_wrapper_type eigen_fft_;

      EigenImpl(const Extent &cextents)
      {
        extents_ = interpret_as::column_major(cextents);
        extents_complex_ = extents_;

        // is nfft in eigen implementation
        n_ = std::accumulate(extents_.begin(),
                             extents_.end(),
                             1,
                             std::multiplies<std::size_t>());

        if (!IsComplex)
        {
          extents_complex_.back() = (extents_.back() / 2 + 1);
        }

        n_complex_ = std::accumulate(extents_complex_.begin(),
                                     extents_complex_.end(),
                                     1,
                                     std::multiplies<size_t>());

        data_size_ = (IsInplaceReal ? 2 * n_complex_ : n_);
        if (!IsInplace) {
          data_complex_size_ = n_complex_;
        }
        else {
          data_complex_size_ = data_size_;
        }
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
        data_ = MemoryAPI::malloc<value_type>(data_size_);
        eigen_data_ = new eigen_map_data_type(data_, data_size_, /*NDim*/ 1);

        if(IsInplace) {
          data_complex_ = reinterpret_cast<ComplexType*>(data_);
        }
        else {
          data_complex_ = MemoryAPI::malloc<ComplexType>(data_complex_size_);
        }

        eigen_data_complex_ = new eigen_map_data_complex_type(data_complex_, data_complex_size_, /*NDim*/ 1);
      }

      void destroy()
      {
        delete eigen_data_;
        delete eigen_data_complex_;

        if(data_)
          MemoryAPI::free(data_);
        data_ = nullptr;

        if(data_complex_ && !IsInplace)
          MemoryAPI::free(data_complex_);
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
        // when inplace-real then alloc'd data is bigger than data to be transferred
        return IsInplaceReal ? n_*sizeof(RealType) : data_size_ * sizeof(value_type);
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
        eigen_fft_ = fft_wrapper_type(); // re-call constructor (can also add opts later here)
        // Plan creation will happen in warmup rounds hopefully
      }
      void init_inverse()
      {
        // Plan creation will happen in warmup rounds hopefully
      }

      void execute_forward()
      {
        eigen_fft_.fwd(*eigen_data_complex_, *eigen_data_);
      }

      void execute_inverse()
      {
        eigen_fft_.inv(*eigen_data_, *eigen_data_complex_);
      }

      // todo: how does reuseplan work? fft.hpp just seems to call init_inverse at
      // different points depending on whether it is set. I'm not sure how this works.

      // Assumes that THostData is a pointer to a value_type
      template <typename THostData>
      void upload(THostData *input)
      {
        if(!IsInplaceReal){
          MemoryAPI::memcpy(data_, input, data_size_);
        } else {
          const std::size_t max_z = (NDim >= 3 ? extents_[NDim-3] : 1);
          const std::size_t max_y = (NDim >= 2 ? extents_[NDim-2] : 1);
          const std::size_t max_x = extents_[NDim-1];
          const std::size_t allocated_x = 2*(extents_[NDim-1]/2+1);

          std::size_t input_index = 0;
          std::size_t data_index = 0;

          for(std::size_t z = 0;z < max_z;++z){
            for(std::size_t y = 0;y < max_y;++y){
              input_index = z*(max_y*max_x) + y*max_x;
              data_index = z*(max_y*allocated_x) + y*allocated_x;
              MemoryAPI::memcpy(data_ + data_index,
                                input + input_index,
                                max_x * sizeof(value_type));
            }
          }
        }
      }

      // Assumes that THostData is a pointer to a value_type
      template <typename THostData>
      void download(THostData *output)
      {
        static_assert(std::is_same<THostData,value_type>::value);
        if(!IsInplaceReal){
          MemoryAPI::memcpy(output, data_, data_size_);
        } else {
          const std::size_t max_z = (NDim >= 3 ? extents_[NDim-3] : 1);
          const std::size_t max_y = (NDim >= 2 ? extents_[NDim-2] : 1);
          const std::size_t max_x = extents_[NDim-1];
          const std::size_t allocated_x = 2*(extents_[NDim-1]/2+1);

          std::size_t output_index = 0;
          std::size_t data_index = 0;

          for(std::size_t z = 0;z < max_z;++z){
            for(std::size_t y = 0;y < max_y;++y){
              output_index = z*(max_y*max_x) + y*max_x;
              data_index = z*(max_y*allocated_x) + y*allocated_x;
              MemoryAPI::memcpy(output+output_index,
                                data_ + data_index,
                                max_x * sizeof(value_type));
            }
          }
        }
      }
    };

    using Inplace_Real = gearshifft::FFT<FFT_Inplace_Real,
                                         FFT_Plan_Not_Reusable,
                                         EigenImpl,
                                         TimerCPU>;

    using Outplace_Real = gearshifft::FFT<FFT_Outplace_Real,
                                          FFT_Plan_Not_Reusable,
                                          EigenImpl,
                                          TimerCPU>;

    using Inplace_Complex = gearshifft::FFT<FFT_Inplace_Complex,
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
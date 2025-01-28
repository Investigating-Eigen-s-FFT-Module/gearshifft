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

namespace gearshifft {
    namespace eigen {
        class EigenOptions : public OptionsDefault {

            public:
                EigenOptions() : OptionsDefault() {}; // todo: add different back-ends as options
        };

        namespace traits {
            template<typename T_Precision=float>
            struct plan {
                using ComplexType = std::complex<T_Precision>;
                using Realtype = float;
                
            }
        }

        struct EigenContext : public ContextDefault<EigenOptions> {
            static const std::string title() {
                // todo: differentiate different back-ends
                return "eigen";
            }

            static std::string get_device_list() {
                // todo: update for multithread support
                std::ostringstream msg;

                msg << "Only single thread CPU supported.\n";

                return msg.str();
            }

            std::string get_used_device_properties() {
                std::ostringstream msg;

                msg << "\"SupportedThreads\"," << 1
                << ",\"UsedThreads\"," << 1
                << ",\"TotalMemory\"," << getMemorySize();

                return msg.str();
            }
        };

        template<typename TFFT,       // see fft.hpp (FFT_Inplace_Real, ...)
                 typename TPrecision, // double
                 size_t NDim // 1 todo: allow 2 and 3 with for loop
                 >
        struct EigenImpl {
            //////////////////////////////////////////////////////////////////////////////////////
            // COMPILE TIME FIELDS

            using Extent = std::array<std::size_t, NDim>;
            using ComplexType = typename traits::plan<TPrecision>::Complextype;
        }
    }
}
#endif /* EIGEN_HPP_ */
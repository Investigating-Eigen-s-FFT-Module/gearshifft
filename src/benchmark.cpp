#include "core/benchmark.hpp"
#include "core/types.hpp"

#include <cstdlib>

// ----------------------------------------------------------------------------
template<typename... Types>
using List = gearshifft::List<Types...>;

#ifdef CUFFT_ENABLED
#include "libraries/cufft/cufft.hpp"

using namespace gearshifft::CuFFT;
using Context           = CuFFTContext;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;

using Precisions        = gearshifft::DefaultPrecisions;
using FFT_Is_Normalized = std::false_type;

#elif defined(CLFFT_ENABLED)
#include "libraries/clfft/clfft.hpp"

using namespace gearshifft::ClFFT;
using Context           = ClFFTContext;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = gearshifft::DefaultPrecisionsWithoutHalfPrecision;
using FFT_Is_Normalized = std::true_type;

#elif defined(FFTW_ENABLED)
#include "libraries/fftw/fftw.hpp"

using namespace gearshifft::fftw;
using Context           = FftwContext;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex >;
using Precisions        = gearshifft::DefaultPrecisionsWithoutHalfPrecision;
using FFT_Is_Normalized = std::false_type;
#elif defined(ROCFFT_ENABLED)
#include "libraries/rocfft/rocfft.hpp"

using namespace gearshifft::RocFFT;
using Context           = RocFFTContext;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex >;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::false_type;
#elif defined(EIGEN_ENABLED)
#include "libraries/eigen/eigen.hpp"

using namespace gearshifft::eigen;
using Context           = EigenContext;
using FFTs              = List</* Inplace_Real, */
                              /* Inplace_Complex, */ // not possible with Eigen API (resize calls for half spectrum stuff on dst breaks it)
                               Outplace_Real,
                               Outplace_Complex >;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::true_type; // todo: I believe this is the case by default but
                                          // it can be changed too?
#endif

// ----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  char reportLevel[] = "BOOST_TEST_REPORT_LEVEL=no";
  putenv(reportLevel);

  int ret = 0;
  try {
    gearshifft::Benchmark<Context> benchmark;

    benchmark.configure(argc, argv);
    ret = benchmark.run<FFT_Is_Normalized, FFTs, Precisions>();

  }catch(const std::runtime_error& e){
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return ret;
}

#include "fatehov_k_gaussian/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace fatehov_k_gaussian {

namespace {

constexpr int kKernelRadius = 1;
constexpr int kKernelSize = 3;
constexpr int kKernelElements = kKernelSize * kKernelSize;
constexpr int kMinPixelValue = 0;
constexpr int kMaxPixelValue = 255;

using Kernel = std::array<float, kKernelElements>;

std::size_t GetPixelIndex(std::size_t y, std::size_t x, std::size_t channel, std::size_t width, std::size_t channels) {
  return ((y * width) + x) * channels + channel;
}

int ClampCoordinate(int coordinate, std::size_t limit) {
  return std::clamp(coordinate, 0, static_cast<int>(limit) - 1);
}

Kernel CreateGaussianKernel(float sigma) {
  Kernel kernel{};
  float sum = 0.0F;

  for (int y = -kKernelRadius; y <= kKernelRadius; ++y) {
    for (int x = -kKernelRadius; x <= kKernelRadius; ++x) {
      const int kernel_y = y + kKernelRadius;
      const int kernel_x = x + kKernelRadius;
      const std::size_t index = static_cast<std::size_t>((kernel_y * kKernelSize) + kernel_x);

      const float distance = static_cast<float>((x * x) + (y * y));
      const float value = std::exp(-distance / (2.0F * sigma * sigma));

      kernel[index] = value;
      sum += value;
    }
  }

  for (float &value : kernel) {
    value /= sum;
  }

  return kernel;
}

uint8_t ApplyKernelToPixel(const Image &src, const Kernel &kernel, std::size_t y, std::size_t x, std::size_t channel) {
  float pixel_value = 0.0F;

  for (int ky = -kKernelRadius; ky <= kKernelRadius; ++ky) {
    const int current_y = ClampCoordinate(static_cast<int>(y) + ky, src.height);

    for (int kx = -kKernelRadius; kx <= kKernelRadius; ++kx) {
      const int current_x = ClampCoordinate(static_cast<int>(x) + kx, src.width);

      const int kernel_y = ky + kKernelRadius;
      const int kernel_x = kx + kKernelRadius;
      const std::size_t kernel_index = static_cast<std::size_t>((kernel_y * kKernelSize) + kernel_x);

      const std::size_t src_index = GetPixelIndex(
          static_cast<std::size_t>(current_y), static_cast<std::size_t>(current_x), channel, src.width, src.channels);

      pixel_value += static_cast<float>(src.data[src_index]) * kernel[kernel_index];
    }
  }

  const int rounded_value = static_cast<int>(std::round(pixel_value));
  return static_cast<uint8_t>(std::clamp(rounded_value, kMinPixelValue, kMaxPixelValue));
}

void ProcessImageRows(const Image &src, Image &dst, const Kernel &kernel,
                      const oneapi::tbb::blocked_range<std::size_t> &range) {
  for (std::size_t y = range.begin(); y < range.end(); ++y) {
    for (std::size_t x = 0; x < src.width; ++x) {
      for (std::size_t channel = 0; channel < src.channels; ++channel) {
        const std::size_t dst_index = GetPixelIndex(y, x, channel, src.width, src.channels);
        dst.data[dst_index] = ApplyKernelToPixel(src, kernel, y, x, channel);
      }
    }
  }
}

void ApplyGaussianFilterTBB(const Image &src, Image &dst, float sigma) {
  const Kernel kernel = CreateGaussianKernel(sigma);

  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<std::size_t>(0, static_cast<std::size_t>(src.height)),
      [&](const oneapi::tbb::blocked_range<std::size_t> &range) { ProcessImageRows(src, dst, kernel, range); });
}

}  // namespace

FatehovKGaussianTBB::FatehovKGaussianTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Image{};
}

bool FatehovKGaussianTBB::ValidationImpl() {
  const auto &input = GetInput();

  if (input.sigma <= 0.0F) {
    return false;
  }

  if (input.image.width == 0 || input.image.height == 0 || input.image.channels == 0) {
    return false;
  }

  const std::size_t expected_size =
      static_cast<std::size_t>(input.image.width) * input.image.height * input.image.channels;

  return input.image.data.size() == expected_size;
}

bool FatehovKGaussianTBB::PreProcessingImpl() {
  const auto &input_image = GetInput().image;

  GetOutput() = Image(input_image.width, input_image.height, input_image.channels);

  return GetOutput().data.size() == input_image.data.size();
}

bool FatehovKGaussianTBB::RunImpl() {
  const auto &input = GetInput();

  ApplyGaussianFilterTBB(input.image, GetOutput(), input.sigma);

  return true;
}

bool FatehovKGaussianTBB::PostProcessingImpl() {
  return !GetOutput().data.empty();
}

}  // namespace fatehov_k_gaussian

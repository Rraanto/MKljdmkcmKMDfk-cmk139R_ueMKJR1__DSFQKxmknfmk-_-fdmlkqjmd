#ifndef COLOR_DISTRIBUTION_H
#define COLOR_DISTRIBUTION_H

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>

struct ColorDistribution {
  float data[8][8][8]; // histogramme
  /**
   *
   * data[8][8][8]: { 8 {8 {8 float}}}
   *
   */
  int nb;

  std::array<uint8_t, 512> counts;

  uint16_t nz_idx[64]; // garder les zeros
  uint8_t nz_val[64];
  uint8_t nz_len;

  ColorDistribution() { reset(); }

  ColorDistribution &operator=(const ColorDistribution &other) = default;

  // met à zero l'histogramme

  void reset();

  void add(cv::Vec3b color);

  void finished();

  float distance(const ColorDistribution &) const;

  static ColorDistribution getColorDistribution(cv::Mat, cv::Point, cv::Point);
};

#endif

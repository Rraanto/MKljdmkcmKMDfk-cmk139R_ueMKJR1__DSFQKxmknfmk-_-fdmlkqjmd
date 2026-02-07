#ifndef COLOR_DISTRIBUTION_H
#define COLOR_DISTRIBUTION_H

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

struct ColorDistribution {
  float data[8][8][8]; // histogramme
  /**
   *
   * data[8][8][8]: { 8 {8 {8 float}}}
   *
   */
  int nb;

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

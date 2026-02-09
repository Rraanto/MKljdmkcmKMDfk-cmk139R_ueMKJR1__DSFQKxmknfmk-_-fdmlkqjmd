#include "color_distribution.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void ColorDistribution::reset() {
  // pour acceder au data comme tableau 1d
  float *data_ptr = &data[0][0][0];
  for (int i = 0; i < 8 * 8 * 8; i++) {
    data_ptr[i] = 0;
  }
  nb = 0;
}

void ColorDistribution::add(cv::Vec3b color) {
  const int b = color[0] / 32;
  const int g = color[1] / 32;
  const int r = color[2] / 32;
  data[r][g][b] += 1.0f;
  nb++;
}

void ColorDistribution::finished() {
  // indique que l'échantillon est fini/complet
  // divise chaque valeur du tableau par le nombre d'échantillons
  if (nb == 0) {
    return;
  }
  const float inv = 1.0f / static_cast<float>(nb);
  float *hist = &data[0][0][0];
  for (int i = 0; i < 8 * 8 * 8; ++i)
    hist[i] *= inv;
}

float ColorDistribution::distance(const ColorDistribution &other) const {

  // pour accéder comme un tableau 1d
  const float *h_1 = &data[0][0][0];
  const float *h_2 = &other.data[0][0][0];

  float total = 0.0f;
  for (int i = 0; i < 8 * 8 * 8; ++i) {
    float diff = h_1[i] - h_2[i];
    diff = diff * diff;

    float sum = h_1[i] + h_2[i];

    total += (sum > 0.0) ? diff / sum : 0.0;
  }
  return total;
}

ColorDistribution ColorDistribution::getColorDistribution(cv::Mat input,
                                                          cv::Point pt1,
                                                          cv::Point pt2) {
  ColorDistribution cd;
  for (int y = pt1.y; y < pt2.y; y++) {
    for (int x = pt1.x; x < pt2.x; x++) {
      cd.add(input.at<cv::Vec3b>(y, x));
    }
  }
  return cd;
}

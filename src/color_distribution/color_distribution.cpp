#include "color_distribution.h"

#include "val_dists.h"
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "immintrin.h"

static inline int idx3(int r, int g, int b) {
  // permet d'obtenir les indices en 1d
  return (r << 6) | (g << 3) | b;
} // r*64 + g*8 + b

void ColorDistribution::reset() {
  // pour acceder au data comme tableau 1d
  float *data_ptr = &data[0][0][0];
  for (int i = 0; i < 8 * 8 * 8; i++) {
    data_ptr[i] = 0;
  }
  counts.fill(0);
  nb = 0;
}

void ColorDistribution::add(cv::Vec3b color) {
  const int b = color[0] / 32;
  const int g = color[1] / 32;
  const int r = color[2] / 32;
  counts[idx3(r, g, b)]++;
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

  // SIMDisation de la multiplication
  float *hist = &data[0][0][0];
  __m128 vinv = _mm_set1_ps(inv);

  for (int i = 0; i < 512; i += 4) {
    __m128 v = _mm_loadu_ps(hist + i);
    v = _mm_mul_ps(v, vinv);
    _mm_storeu_ps(hist + i, v);
  }
}

float ColorDistribution::distance(const ColorDistribution &other) const {

  float total = 0.0f;
  for (int i = 0; i < 512; i++) {
    total += TAB_DIST[counts[i]][other.counts[i]];
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
  cd.finished();
  return cd;
}

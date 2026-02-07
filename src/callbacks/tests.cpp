#include "tests.h"

#include <iostream>

#include "color_distribution.h"

using namespace cv;
using namespace std;

void calcul_distance_gauche_droite(const Mat &img_input, const Point &top_left,
                                   const Point &bottom_center,
                                   const Point &middle_top,
                                   const Point &bottom_right) {
  /*
   * Calcule la distance entre couleurs d'une partie "gauche" et "droite"
   * définies par des points d'encrages bottom_center, middle_top,
   * bottom_right, top_left
   */
  ColorDistribution left_hist;
  left_hist =
      left_hist.getColorDistribution(img_input, top_left, bottom_center);
  left_hist.finished();

  ColorDistribution right_hist;
  right_hist =
      right_hist.getColorDistribution(img_input, middle_top, bottom_right);
  right_hist.finished();

  float d = left_hist.distance(right_hist);
  cout << "Distance de couleurs: " << d << endl;
}

void test_blocs_nb_et_distance(
    Mat &img_input, const Mat &frozen, const Rect &roiGH, const Rect &roiGB,
    const Rect &roiDH, const Rect &roiDB, const Mat &noir, const Mat &blanc,
    bool &config_1, const Point &top_left, const Point &bottom_center,
    const Point &middle_top, const Point &bottom_right) {
  /*
   * effectue le test sur l'image partiulière de blocs noir et blancs
   */
  if (frozen.empty())
    return;

  img_input = frozen.clone();

  blanc.copyTo(img_input(roiGH));
  blanc.copyTo(img_input((config_1) ? roiDB : roiDH));

  noir.copyTo(img_input(roiGB));
  noir.copyTo(img_input((config_1) ? roiDH : roiDB));

  config_1 = !config_1;

  calcul_distance_gauche_droite(img_input, top_left, bottom_center, middle_top,
                                bottom_right);
}

/*
 * Set up initial avec mini application de capture vidéo
 */

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "callbacks/reconnaissance.h"
#include "callbacks/tests.h"
#include "color_distribution.h"
#include <vector>

#define FPS 30

using namespace cv;
using namespace std;

int main(int ragc, char **argv) {
  Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;

  VideoCapture *pCap = nullptr;
  const int width = 640;
  const int height = 480;
  const int size = 50;

  // ouvre la cam
  pCap = new VideoCapture(0);
  if (!pCap->isOpened()) {
    cout << "Ne peux pas ouvrir la caméera";
    return 1;
  }

  /*
   * Setup de l'image sur lequel afficher le flux vidéo
   */

  pCap->set(CAP_PROP_FRAME_WIDTH, 640);
  pCap->set(CAP_PROP_FRAME_HEIGHT, 480);

  (*pCap) >> img_input;
  if (img_input.empty()) {
    cout << "Problème avec la caméra (ouverte)";
    return 1;
  }

  Point pt1(width / 2 - size / 2, height / 2 - size / 2);
  Point pt2(width / 2 + size / 2, height / 2 + size / 2);

  namedWindow("input", 1);
  imshow("input", img_input);

  bool freeze = false;
  Mat frozen; // pour restaurer l'image précédente
              // quand on teste les rectangles NB

  /*
   * Pour stocker histogrammes fond et objets
   * all_col_hists[0] = fond
   * all_col_hists[1] = objet 1
   * all_col_hists[2] = objet 2
   * ...
   */
  std::vector<std::vector<ColorDistribution>> all_col_hists;
  all_col_hists.resize(2);

  /* objet courant pour l'acquisition (mode B) */
  int obj_courant = 1;

  /*
   * Interrupteurs pour switcher entre différents "modes"
   */
  bool config_1 = true; // pour alterner entre les configurations
  bool reco = false;

  /*
   * Couleurs pour la visualisation:
   * colors[k] correspond à all_col_hists[k]
   */
  std::vector<Vec3b> colors;
  colors.push_back(Vec3b(0, 0, 0));   // fond: noir
  colors.push_back(Vec3b(0, 0, 255)); // objet 1: rouge

  /* palette simple (BGR) pour les objets supplémentaires */
  const std::vector<Vec3b> palette = {
      Vec3b(0, 0, 255),    // rouge
      Vec3b(0, 255, 0),    // vert
      Vec3b(255, 0, 0),    // bleu
      Vec3b(0, 255, 255),  // jaune
      Vec3b(255, 0, 255),  // magenta
      Vec3b(255, 255, 0),  // cyan
      Vec3b(255, 255, 255) // blanc
  };

  /*
   * Boucle principale
   */
  while (true) {
    // 1000 ms -> FPS images
    // ? ms    <- 1 image
    // => 1000 / FPS
    char c = (char)waitKey(1000 / FPS);

    if (pCap != nullptr && !freeze) {
      (*pCap) >> img_input; // recupere le flux
      cv::flip(img_input, img_input, 1);
    }

    if (c == 27 || c == 'q')
      break;

    if (c == 'f') { // geler l'image
      freeze = !freeze;
      if (freeze)
        frozen = img_input.clone();
    }

    if (c == 'v') {
      calcul_distance_gauche_droite(img_input);
    }

    // Histogrammes du fond (all_col_hists[0])
    if (c == 'b') {
      calcul_histogrammes_fond(all_col_hists[0], img_input);
    }

    // acquisition histogramme de l'objet courant
    // (all_col_hists[current_object])
    if (c == 'a') {
      ajout_histogramme_objet(all_col_hists[obj_courant], img_input, pt1, pt2);
      cout << "[a] objet courant = " << obj_courant
           << " / nb hist = " << all_col_hists[obj_courant].size() << endl;
    }

    // creer un nouvel objet (option B)
    if (c == 'n') {
      all_col_hists.push_back(std::vector<ColorDistribution>());
      obj_courant = all_col_hists.size() - 1;

      if (colors.size() < all_col_hists.size()) {
        int idx = colors.size() - 1; // 0 est fond; objets commencent a 1
        Vec3b next_color = palette[(idx - 1) % palette.size()];
        colors.push_back(next_color);
      }

      cout << "[n] nouvel objet = " << obj_courant
           << " / nb objets = " << (int)all_col_hists.size() - 1 << endl;
    }

    if (c == 'x' && freeze) {
      test_blocs_nb_et_distance(img_input, frozen, config_1);
    }

    if (c == 'r') {
      if (all_col_hists[0].empty()) {
        cout << "Pas d'histogrammes fond" << endl;
      } else {
        bool has_object = false;
        for (size_t k = 1; k < all_col_hists.size(); ++k) {
          if (!all_col_hists[k].empty()) {
            has_object = true;
            break;
          }
        }
        if (!has_object) {
          cout << "Pas d'histogrammes objets" << endl;
        } else {
          reco = !reco;
        }
      }
    }

    Mat out = img_input;

    // choix d'image à afficher selon mode reco ou non
    if (reco) {
      Mat gray;
      cvtColor(img_input, gray, COLOR_BGR2GRAY);

      Mat reco_img = recoObject(img_input, all_col_hists, colors, 16);

      cvtColor(gray, img_input, COLOR_GRAY2BGR);
      out = 0.5 * reco_img + 0.5 * img_input;
    } else {
      cv::rectangle(img_input, pt1, pt2, Scalar({255.0, 255.0, 255.0}), 1);
    }

    imshow("input", out);
  }

  return 0;
}

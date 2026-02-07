#include "reconnaissance.h"

#include <algorithm>
#include <iostream>
#include <limits>

using namespace cv;
using namespace std;

void
ajout_histogramme_objet(std::vector<ColorDistribution> &col_hists_object,
                        const Mat &img_input, const Point &haut_gauche,
                        const Point &bas_droite) {
  /*
   * Ajoute l'histogramme actuel d'un objet délimité par une zone
   * rectangulaire définie par deux points d'encrages (haut_gauche et
   * bas_droite)
   */
  ColorDistribution obj = ColorDistribution::getColorDistribution(
      img_input, haut_gauche, bas_droite);

  obj.finished();
  col_hists_object.push_back(obj);
  cout << "[a] histogramme objets = " << col_hists_object.size() << endl;
}

void calcul_histogrammes_fond(std::vector<ColorDistribution> &col_hists,
                              const Mat &img_input) {
  /*
   * calcule les histogrammes du fond
   * découpé en blocs de taille 128x128
   * et les ajoute à un vecteur déjà initialisé (passé en paramètre)
   */
  col_hists.clear();
  const int bbloc = 128;

  for (int y = 0; y <= img_input.rows - bbloc; y += bbloc) {
    for (int x = 0; x <= img_input.cols - bbloc; x += bbloc) {
      Point p1(x, y);
      Point p2(x + bbloc, y + bbloc);
      ColorDistribution cd =
          ColorDistribution::getColorDistribution(img_input, p1, p2);

      cd.finished();
      col_hists.push_back(cd);
    }
  }

  int nb_hists_background = (int)col_hists.size();
  cout << "[b] histogramme fonds : " << nb_hists_background << endl;
}

static float minDistance(const ColorDistribution &h,
                         const std::vector<ColorDistribution> &hists) {
  float plus_proche = std::numeric_limits<float>::infinity();
  for (const ColorDistribution &x : hists) {
    plus_proche = std::min(plus_proche, h.distance(x));
  }
  return plus_proche;
}

Mat recoObject(Mat input, const std::vector<ColorDistribution> &col_hists,
               const std::vector<ColorDistribution> &col_hists_object,
               const std::vector<Vec3b> &colors, const int bloc) {

  /*
   * Assigne les couleurs déterminées aux régions détectés comme "objet" ou
   * backgroudn
   *
   * ATTENTION: modifie les vecteurs col_hists, col_hists_object (normalisation)
   *
   */
  Mat output = input.clone();

  for (int y = 0; y <= input.rows - bloc; y += bloc) {
    for (int x = 0; x <= input.cols - bloc; x += bloc) {
      Point pt1(x, y);
      Point pt2(x + bloc, y + bloc);

      ColorDistribution h =
          ColorDistribution::getColorDistribution(input, pt1, pt2);

      float d_bg =
          minDistance(h, col_hists); // distance minimale contre les fonds
      float d_obj = minDistance(
          h, col_hists_object); // distance minimale contre les objets

      // assignation de la couleur selon tyep
      const Vec3b &c = (d_bg < d_obj) ? colors[0] : colors[1];

      Rect r(x, y, bloc, bloc);
      output(r).setTo(Scalar(c[0], c[1], c[2]));
    }
  }
  return output;
}

/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;
  std::normal_distribution<double> norm_x(x,std[0]);
  std::normal_distribution<double> norm_y(y,std[1]);
  std::normal_distribution<double> norm_theta(theta,std[2]);
  for (int i=0; i < num_particles; ++i) {
    double noisy_x = norm_x(gen);
    double noisy_y = norm_y(gen);
    double noisy_theta = norm_theta(gen);
    //std::cout << "x" << noisy_x << "y" << noisy_y << std::endl;
    Particle p = { p.id = i, p.x = noisy_x, p.y = noisy_y, p.theta = noisy_theta, p.weight=1};
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

}

void ParticleFilter::dataAssociation(Particle &particle,
                                     Map map_landmarks, vector<double> sense_x,
                                     vector<double> sense_y) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
   vector<int> landmark_assocs;
   // Iterate over each transformed observation of the particle and find the closest
   // landmark for each observation
   //std::cout << "sense x" << sense_x;
   for (int i=0; i<sense_x.size(); ++i) {
     std::cout << "Trying to find landmark for observation " << i << std::endl;
     int min_index = -100000;
     int min_distance = 100000;
     for (int j=0; j<map_landmarks.landmark_list.size(); ++j) {
       double distance = dist(sense_x[i], sense_y[i],
                              map_landmarks.landmark_list[j].x_f,
                              map_landmarks.landmark_list[j].y_f );
       if (distance < min_distance) {
         min_index = map_landmarks.landmark_list[j].id_i;
         min_distance = distance;
       }
     }
     landmark_assocs.push_back(min_index);
   }
   SetAssociations(particle, landmark_assocs, sense_x, sense_y);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // 1. Transform landmark observations into map coordinates
  for (int i=0; i<particles.size(); ++i) {
    std::cout << "Particle " << i << std::endl;
    // Hold transformed x and y observations
    vector<double> tx_obs;
    vector<double> ty_obs;
    for (int j=0; j<observations.size(); ++j) {
      std::cout << "Observation: " << j << std::endl;

      Particle p = particles[i];
      LandmarkObs obs = observations[j];
      // Transform car observations into particle (world) coordinates
      double world_x = obs.x * cos(p.theta) - obs.y*sin(p.theta) + p.x;
      double world_y = obs.x * sin(p.theta) - obs.y*cos(p.theta) + p.y;
      tx_obs.push_back(world_x);
      ty_obs.push_back(world_y);
      //LandmarkObs t_obs;
      //t_obs.id = obs.id;
      //t_obs.x = obs.x * cos(p.theta) - obs.y*sin(p.theta) + p.x;
      //t_obs.y = obs.x * sin(p.theta) - obs.y*cos(p.theta) + p.y;
      //t_observations.push_back(t_obs);
    }
    dataAssociation(particles[i], map_landmarks, tx_obs, ty_obs);

    // Update associations

  }
  // 2. Associate transformed landmark observations with nearest landmarks


}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

#include<string>
#include<vector>
#include "Halide.h"
#include "halide_image_io.h"

void get_image_names_and_labels(std::string label_file_name,
                                std::vector<std::string> &image_names,
                                std::vector<int> &image_labels);

void get_reference_labels(std::string reference_file_name,
                          std::vector<int>& labels,
                          std::vector<float>& scores);

void load_cifar_batch(std::string bin_path, int batch_size,
                      int index, Image<float> &batch,
                      Image<int> &image_labels);

void load_cifar_batch_random(std::string bin_path, int batch_size,
                             Image<float> &batch,
                             Image<int> &image_labels);

int get_cifar_num_images(std::string bin_path);

void load_imagenet_batch(std::vector<std::string> &image_names,
                         std::string image_dir, size_t index,
                         bool subtract_mean, Image<float> &batch);

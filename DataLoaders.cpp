#include<iostream>
#include<fstream>
#include<iterator>
#include<assert.h>
#include<random>

#include "Halide.h"
#include "halide_image_io.h"


void get_image_names_and_labels(std::string label_file_name,
                                std::vector<std::string> &image_names,
                                std::vector<int> &image_labels) {
    std::ifstream label_file(label_file_name);
    std::string line;

    if (!label_file) {
        std::cout << "Could not open labeled data file: " << label_file_name << std::endl;
        exit(1);
    }

    while (std::getline(label_file, line)) {
        size_t space_loc = line.find_first_of(' ');
        image_names.push_back(line.substr(0, space_loc));
        int label = std::stoi(line.substr(space_loc + 1));
        image_labels.push_back(label);
    }
}

void get_reference_labels(std::string reference_file_name,
                          std::vector<int>& labels,
                          std::vector<float>& scores) {
    std::ifstream reference_file(reference_file_name);
    std::string line;

    if (!reference_file) {
        std::cout << "Could not open reference solution file: " << reference_file_name << std::endl;
        exit(1);
    }

    while (std::getline(reference_file, line)) {
        std::string tmp1, tmp2;
        std::istringstream iss(line);
        int label;
        float score;
        iss >> tmp1 >> label >> tmp2 >> score;
        labels.push_back(label);
        scores.push_back(score);
    }
}

int get_cifar_num_images(std::string bin_path) {
    std::ifstream file(bin_path, std::ifstream::binary | std::ifstream::ate);

    if (!file) {
        std::cout << "Cannot open CIFAR data file: " << bin_path << std::endl;
        exit(1);
    }

    if (file.is_open()) {
        int n_rows = 32;
        int n_cols = 32;
        size_t bytes_per_image = 2 + n_rows * n_cols * 3;
        size_t num_bytes_in_file = file.tellg();
        assert(num_bytes_in_file % bytes_per_image == 0);

        return num_bytes_in_file / bytes_per_image;
    } else {
        return 0;
    }
}

void load_cifar_batch_random(std::string bin_path, int batch_size,
                             Halide::Image<float> mean,
                             Halide::Image<float> &batch,
                             Halide::Image<int> &image_labels) {

    std::ifstream file(bin_path, std::ifstream::binary | std::ifstream::ate);

    if (!file) {
        std::cout << "Cannot open CIFAR data file: " << bin_path << std::endl;
        exit(1);
    }

    if (file.is_open()) {
        int n_rows = 32;
        int n_cols = 32;
        size_t bytes_per_image = 2 + n_rows * n_cols * 3;
        size_t num_bytes_in_file = file.tellg();
        assert(num_bytes_in_file % bytes_per_image == 0);

        int number_of_images = num_bytes_in_file/bytes_per_image;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, number_of_images - 1);

        for(int i = 0; i < batch_size; ++i) {
            // Draw a random index
            int index = dis(gen);
            file.seekg(bytes_per_image * index);
            unsigned char label = 0;
            // Ignore the super class label
            file.read((char*) &label, sizeof(label));
            // Get the fine grained label
            file.read((char*) &label, sizeof(label));
            image_labels(i) = (int)label;
            for(int ch = 0; ch < 3; ++ch) {
                for(int c = 0; c < n_cols; ++c) {
                    for(int r = 0; r < n_rows; ++r) {
                        unsigned char pix = 0;
                        file.read((char*) &pix, sizeof(pix));
                        batch(r, c, ch, i) = (float) pix - mean(r, c, ch);
                    }
                }
            }
        }
    }
}

void load_cifar_batch(std::string bin_path, int batch_size,
                      int index, Halide::Image<float> mean,
                      Halide::Image<float> &batch,
                      Halide::Image<int> &image_labels) {

    std::ifstream file(bin_path, std::ifstream::binary | std::ifstream::ate);

    if (!file) {
        std::cout << "Cannot open CIFAR data file: " << bin_path << std::endl;
        exit(1);
    }

    if (file.is_open()) {
        int n_rows = 32;
        int n_cols = 32;
        size_t bytes_per_image = 2 + n_rows * n_cols * 3;
        size_t num_bytes_in_file = file.tellg();
        assert(num_bytes_in_file % bytes_per_image == 0);

        int number_of_images = num_bytes_in_file/bytes_per_image;
        assert(index < number_of_images);
        file.seekg(bytes_per_image * index);

        int num_images_to_read = std::min(number_of_images - index, batch_size);

        for(int i = 0; i < num_images_to_read; ++i) {
            unsigned char label = 0;
            // Ignore the super class label
            file.read((char*) &label, sizeof(label));
            // Get the fine grained label
            file.read((char*) &label, sizeof(label));
            image_labels(i) = (int)label;
            for(int ch = 0; ch < 3; ++ch) {
                for(int c = 0; c < n_cols; ++c) {
                    for(int r = 0; r < n_rows; ++r) {
                        unsigned char pix = 0;
                        file.read((char*) &pix, sizeof(pix));
                        batch(r, c, ch, i) = (float) pix - mean(r, c, ch);
                    }
                }
            }
        }
    }
}

void compute_cifar_mean(std::string bin_path, Halide::Image<float> &mean) {

    std::ifstream file(bin_path, std::ifstream::binary | std::ifstream::ate);

    if (!file) {
        std::cout << "Cannot open CIFAR data file: " << bin_path << std::endl;
        exit(1);
    }

    if (file.is_open()) {
        int n_rows = 32;
        int n_cols = 32;
        size_t bytes_per_image = 2 + n_rows * n_cols * 3;
        size_t num_bytes_in_file = file.tellg();
        assert(num_bytes_in_file % bytes_per_image == 0);

        for(int ch = 0; ch < 3; ++ch) {
            for(int c = 0; c < n_cols; ++c) {
                for(int r = 0; r < n_rows; ++r) {
                    mean(r, c, ch) = 0.0f;
                }
            }
        }

        int number_of_images = num_bytes_in_file/bytes_per_image;
        file.seekg(0);

        float scale = 1.0f/(number_of_images);
        for(int i = 0; i < number_of_images; ++i) {
            unsigned char label = 0;
            // Ignore the super class label
            file.read((char*) &label, sizeof(label));
            // Get the fine grained label
            file.read((char*) &label, sizeof(label));
            for(int ch = 0; ch < 3; ++ch) {
                for(int c = 0; c < n_cols; ++c) {
                    for(int r = 0; r < n_rows; ++r) {
                        unsigned char pix = 0;
                        file.read((char*) &pix, sizeof(pix));
                        mean(r, c, ch) += (float) pix * scale;
                    }
                }
            }
        }
    }
}

void load_imagenet_batch(std::vector<std::string> &image_names,
                         std::string image_dir, size_t index,
                         bool subtract_mean, Halide::Image<float> &batch) {
    assert(batch.dimensions() == 4);
    size_t total_images = image_names.size();
    assert(index < total_images);
    size_t batch_size = batch.extent(3);

    size_t upper_bound = std::min(batch_size + index, total_images);
    size_t id = 0;

    for (size_t i = index; i < upper_bound; i++) {
        std::string image_path = image_dir + "/" + image_names[i];

        Halide::Image<uint8_t> img = Halide::Tools::load_image(image_path);
        assert(batch.extent(0) == img.extent(0) &&
               batch.extent(1) == img.extent(1));

        if (img.dimensions() == 3) {
            float r_mean = subtract_mean? 122.67891434f : 0.0f;
            for (int h = 0; h < img.extent(1); h++) {
                for (int w = 0; w < img.extent(0); w++) {
                    batch(w, h, 2, id) = img(w, h, 0) - r_mean;
                }
            }
            float b_mean = subtract_mean? 116.66876762f : 0.0f;
            for (int h = 0; h < img.extent(1); h++) {
                for (int w = 0; w < img.extent(0); w++) {
                    batch(w, h, 1, id) = img(w, h, 1) - b_mean;
                }
            }
            float g_mean = subtract_mean? 104.00698793f : 0.0f;
            for (int h = 0; h < img.extent(1); h++) {
                for (int w = 0; w < img.extent(0); w++) {
                    batch(w, h, 0, id) = img(w, h, 2) - g_mean;
                }
            }
        } else if (img.dimensions() == 2) {
            for (int c = 0; c < batch.extent(2); c++) {
                for (int h = 0; h < img.extent(1); h++) {
                    for (int w = 0; w < img.extent(0); w++) {
                        batch(w, h, c, id) = img(w, h);
                    }
                }
            }
        } else {
            assert(0);
        }
        id++;
    }
}

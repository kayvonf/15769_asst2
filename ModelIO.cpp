//#include <caffe/caffe.hpp>
#include <iostream>
#include <fstream>
#include "ModelIO.h"

//using namespace caffe;
using namespace Halide;

void save_model_to_disk(std::string weight_file_name, Weights &weights) {
    std::ofstream ofs;
    ofs.open(weight_file_name, std::ofstream::out | std::ofstream::trunc |
                               std::ofstream::binary);

    size_t num_weights = weights.size();
    ofs.write(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
    for (auto &w: weights) {
        size_t name_len = w.first.size();
        ofs.write(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        ofs.write(const_cast<char*>(w.first.c_str()), name_len);

        size_t num_params = w.second.size();
        ofs.write(reinterpret_cast<char*>(&num_params), sizeof(num_params));

        for (size_t i = 0; i < w.second.size(); i++) {
            Image<float> param = w.second[i];
            int dims = param.dimensions();
            ofs.write(reinterpret_cast<char*>(&dims), sizeof(dims));
            switch (param.dimensions()) {
                case 1: {
                    int d0 = param.extent(0);
                    ofs.write(reinterpret_cast<char*>(&d0), sizeof(d0));
                    for (int i = 0; i < param.extent(0); i++) {
                        ofs.write(reinterpret_cast<char*>(&param(i)), sizeof(float));
                    }
                    break;
                }
                case 2: {
                    int d0 = param.extent(0);
                    ofs.write(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1 = param.extent(1);
                    ofs.write(reinterpret_cast<char*>(&d1), sizeof(d1));
                    for (int i = 0; i < param.extent(0); i++) {
                        for (int j = 0; j < param.extent(1); j++) {
                            ofs.write(reinterpret_cast<char*>(&param(i, j)), sizeof(float));
                        }
                    }
                    break;
                }
                case 3: {
                    int d0 = param.extent(0);
                    ofs.write(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1 = param.extent(1);
                    ofs.write(reinterpret_cast<char*>(&d1), sizeof(d1));
                    int d2 = param.extent(2);
                    ofs.write(reinterpret_cast<char*>(&d2), sizeof(d2));
                    for (int i = 0; i < param.extent(0); i++) {
                        for (int j = 0; j < param.extent(1); j++) {
                            for (int k = 0; k < param.extent(2); k++) {
                                ofs.write(reinterpret_cast<char*>(&param(i, j, k)), sizeof(float));
                            }
                        }
                    }
                    break;
                }
                case 4: {
                    int d0 = param.extent(0);
                    ofs.write(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1 = param.extent(1);
                    ofs.write(reinterpret_cast<char*>(&d1), sizeof(d1));
                    int d2 = param.extent(2);
                    ofs.write(reinterpret_cast<char*>(&d2), sizeof(d2));
                    int d3 = param.extent(3);
                    ofs.write(reinterpret_cast<char*>(&d3), sizeof(d3));
                    for (int i = 0; i < param.extent(0); i++) {
                        for (int j = 0; j < param.extent(1); j++) {
                            for (int k = 0; k < param.extent(2); k++) {
                                for (int l = 0; l < param.extent(3); l++) {
                                    ofs.write(reinterpret_cast<char*>(&param(i, j, k, l)), sizeof(float));
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

void load_model_from_disk(std::string weight_file_name, Weights &weights) {
    std::ifstream ifs;
    ifs.open(weight_file_name, std::ifstream::in | std::ofstream::binary);

    size_t num_weights;
    ifs.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
    for (size_t w = 0; w < num_weights; w++) {
        size_t name_len;
        ifs.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string layer_name;
        layer_name.resize(name_len);
        ifs.read(const_cast<char*>(&layer_name[0]), name_len);

        size_t num_params;
        ifs.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

        for (size_t i = 0; i < num_params; i++) {
            int dims;
            ifs.read(reinterpret_cast<char*>(&dims), sizeof(dims));
            switch (dims) {
                case 1: {
                    int d0;
                    ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
                    Image<float> param(d0);
                    for (int i = 0; i < d0; i++) {
                        ifs.read(reinterpret_cast<char*>(&param(i)), sizeof(float));
                    }
                    weights[layer_name].push_back(param);
                    break;
                }
                case 2: {
                    int d0;
                    ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1;
                    ifs.read(reinterpret_cast<char*>(&d1), sizeof(d1));
                    Image<float> param(d0, d1);
                    for (int i = 0; i < d0; i++) {
                        for (int j = 0; j < d1; j++) {
                            ifs.read(reinterpret_cast<char*>(&param(i, j)), sizeof(float));
                        }
                    }
                    weights[layer_name].push_back(param);
                    break;
                }
                case 3: {
                    int d0;
                    ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1;
                    ifs.read(reinterpret_cast<char*>(&d1), sizeof(d1));
                    int d2;
                    ifs.read(reinterpret_cast<char*>(&d2), sizeof(d2));
                    Image<float> param(d0, d1, d2);
                    for (int i = 0; i < d0; i++) {
                        for (int j = 0; j < d1; j++) {
                            for (int k = 0; k < d2; k++) {
                                ifs.read(reinterpret_cast<char*>(&param(i, j, k)), sizeof(float));
                            }
                        }
                    }
                    weights[layer_name].push_back(param);
                    break;
                }
                case 4: {
                    int d0;
                    ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
                    int d1;
                    ifs.read(reinterpret_cast<char*>(&d1), sizeof(d1));
                    int d2;
                    ifs.read(reinterpret_cast<char*>(&d2), sizeof(d2));
                    int d3;
                    ifs.read(reinterpret_cast<char*>(&d3), sizeof(d3));
                    Image<float> param(d0, d1, d2, d3);
                    for (int i = 0; i < d0; i++) {
                        for (int j = 0; j < d1; j++) {
                            for (int k = 0; k < d2; k++) {
                                for (int l = 0; l < d3; l++) {
                                    ifs.read(reinterpret_cast<char*>(&param(i, j, k, l)), sizeof(float));
                                }
                            }
                        }
                    }
                    weights[layer_name].push_back(param);
                    break;
                }
            }
        }
    }
}

/*
template<typename T>
Image<T> convert_blob_to_image(const Blob<T> &b) {
    switch(b.shape().size()) {
        case 1:
            {
                Image<T> img(b.shape(0));
                for (int i = 0; i < b.shape(0); i++) {
                    img(i) = b.data_at(i, 0, 0, 0);
                }
                return img;
            }
        case 2:
            {
                Image<T> img(b.shape(1), b.shape(0));
                for (int i = 0; i < b.shape(0); i++) {
                    for (int j = 0; j < b.shape(1); j++) {
                        img(j, i) = b.data_at(i, j, 0, 0);
                    }
                }
                return img;
            }
        case 3:
            {
                Image<T> img(b.shape(2), b.shape(1), b.shape(0));
                for (int i = 0; i < b.shape(0); i++) {
                    for (int j = 0; j < b.shape(1); j++) {
                        for (int k = 0; k < b.shape(2); k++) {
                            img(k, j, i) = b.data_at(i, j, k, 0);
                        }
                    }
                }
                return img;
            }
        case 4:
            {
                Image<T> img(b.shape(3), b.shape(2), b.shape(1), b.shape(0));
                for (int i = 0; i < b.shape(0); i++) {
                    for (int j = 0; j < b.shape(1); j++) {
                        for (int k = 0; k < b.shape(2); k++) {
                            for (int l = 0; l < b.shape(3); l++) {
                                img(l, k, j, i) = b.data_at(i, j, k, l);
                            }
                        }
                    }
                }
                return img;
            }
        default:
            std::cout <<
                "Cannot handle a blob with more than 4 dimensions" << std::endl;
            exit(-1);
    }
    return Image<T>();
}

void display_network_info(Net<float> &net) {
    std::cout << "Num layer names:" << net.layer_names().size() << std::endl;
    for (auto &name: net.layer_names()) {
        std::cout << name << std::endl;
    }
    std::cout << "Num blobs:" << net.blob_names().size() << std::endl;
    for (auto &name: net.blob_names()) {
        std::cout << name << std::endl;
    }
}

void load_weights_caffe(char *first_arg,
                        std::string model_file_path,
                        std::string weight_file_path,
                        Weights &weights) {
    // Caffe requires google logging to be initialized
    ::google::InitGoogleLogging(first_arg);
    // Load the network
    Net<float> net(model_file_path, TEST);
    net.CopyTrainedLayersFrom(weight_file_path);
    // Print network information
    // display_network_info(net);
    // Convert caffe blobs into Halide images and populate them
    // into the map of weights
    for (size_t i = 0; i < net.layers().size(); i++) {
        for (auto &b: (*net.layers()[i]).blobs()) {
            weights[net.layer_names()[i]].push_back(convert_blob_to_image(*b));
        }
    }
}
*/

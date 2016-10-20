#include <string>
#include <map>
#include "Halide.h"

typedef std::map<std::string, std::vector<Halide::Image<float>>> Weights;

void load_weights_caffe(char *first_arg,
                        std::string model_file_path,
                        std::string weight_file_path,
                        Weights& weights);

void save_model_to_disk(std::string model_path, Weights &weights);
void load_model_from_disk(std::string weight_file_name, Weights &weights);

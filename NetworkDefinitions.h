#include<map>
#include<string>
#include "Layers.h"
#include "ModelIO.h"


template<typename T>
void copy_image(Image<T> &src, Image<T> &dst) {
    assert(src.dimensions() == dst.dimensions());
    switch(src.dimensions()) {
        case 1:
            assert(src.extent(0) == dst.extent(0));
            for (int i = 0; i < src.extent(0); i++) {
                dst(i) = src(i);
            }
            break;
        case 2:
            assert(src.extent(0) == dst.extent(0) &&
                   src.extent(1) == dst.extent(1));
            for (int i = 0; i < src.extent(0); i++) {
                for (int j = 0; j < src.extent(1); j++) {
                    dst(i, j) = src(i, j);
                }
            }
            break;
        case 3:
            for (int i = 0; i < src.extent(0); i++) {
                for (int j = 0; j < src.extent(1); j++) {
                    for (int k = 0; k < src.extent(2); k++) {
                        dst(i, j, k) = src(i, j, k);
                    }
                }
            }
            break;
        case 4:
            for (int i = 0; i < src.extent(0); i++) {
                for (int j = 0; j < src.extent(1); j++) {
                    for (int k = 0; k < src.extent(2); k++) {
                        for (int l = 0; l < src.extent(3); l++) {
                            dst(i, j, k, l) = src(i, j, k, l);
                        }
                    }
                }
            }
            break;
    }
}

class Network {
    public:
        Network() {}
        virtual ~Network() {
            for (auto l: layers) {
                delete l.second;
            }
        }

        std::map<std::string, Layer*> layers;
        std::vector<std::pair<std::string, std::string>> sub_pipeline_end_points;

        virtual void initialize_weights(Weights &weights);
        virtual void extract_weights(Weights &weights);
        virtual void define_forward(int batch_size, int data_width,
                                    int data_height) = 0;

        virtual void define_backward(Image<int> labels);
        virtual void update_weights();
        virtual void display_layer_shapes();
};

/*
 * Network::initialize_weights --
 *
 * Load weights (already loaded into DRAM) into the network
 */
void Network::initialize_weights(Weights &weights) {
    for (auto &l: layers) {
        if (l.second->params.size() > 0) {
            assert(weights[l.first].size() == l.second->params.size());
            for (size_t p = 0; p < weights[l.first].size(); p++) {
                copy_image(weights[l.first][p], l.second->params[p]);
            }
        }
    }
}

/*
 * Network::extract_weights --
 *
 * Copy weights from network into weights struct for external use or I/O.
 */
void Network::extract_weights(Weights& weights) {
    for (auto &l: layers) {
        weights[l.first] = std::vector<Image<float>>();
        if (l.second->params.size() > 0) {
            for (size_t p = 0; p < l.second->params.size(); p++) {
                Image<float> img(l.second->params[p]);
                weights[l.first].push_back(img);
            }
        }
    }
}

/*
 * Network::display_layer_shapes --
 *
 * Pretty-print the dimensions of the output (activation volume) for a
 * network
 */
void Network::display_layer_shapes() {
    for (auto &l: layers) {
        std::cout << l.first << " [";
        switch (l.second->out_dims()) {
            case 1:
                std::cout << l.second->out_dim_size(0) << "]" << std::endl;
                break;
            case 2:
                std::cout << l.second->out_dim_size(0) << "," <<
                             l.second->out_dim_size(1) << "]" << std::endl;
                break;
            case 3:
                std::cout << l.second->out_dim_size(0) << "," <<
                             l.second->out_dim_size(1) << "," <<
                             l.second->out_dim_size(2) << "]" << std::endl;
                break;
            case 4:
                std::cout << l.second->out_dim_size(0) << "," <<
                             l.second->out_dim_size(1) << "," <<
                             l.second->out_dim_size(2) << "," <<
                             l.second->out_dim_size(3) << "]" << std::endl;
                break;
        }
    }
}

// virtual: must be defined by subclass
void Network::define_backward(Image<int> labels) {}

void Network::update_weights() {}


class Vgg: public Network {
    public:
        Vgg() : Network() {}
        void define_forward(int batch_size, int data_width, int data_height);
};

/*
 * Vgg::define_forward --
 *
 * This function builds the topology of the VGG network by
 * instantiating the appropriate "Layer" classes and writing up their
 * inputs and outputs
 */
void Vgg::define_forward(int batch_size, int data_width, int data_height) {

    // Network structure
    // input -> conv1_1 -> relu1_1 -> conv1_2 -> relu1_2 -> pool1 ->
    // conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> pool2 ->
    // conv3_1 -> relu3_1 -> conv3_2 -> relu3_2 -> conv3_3 -> relu3_3 -> pool3 ->
    // conv4_1 -> relu4_1 -> conv4_2 -> relu4_2 -> conv4_3 -> relu4_3 -> pool4 ->
    // conv5_1 -> relu5_1 -> conv5_2 -> relu5_2 -> conv5_3 -> relu5_3 -> pool5 ->
    // fc6-> relu6 -> drop6-> fc7 -> relu7 -> drop7 -> fc8 -> prob

    std::vector<Layer*> layer_list;

    int channels = 3;

    Image<float> input(data_width, data_height, channels, batch_size);

    DataLayer *d_layer = new DataLayer("input", input);
    layer_list.push_back(d_layer);

    int num_filters_1 = 64;
    int filter_width = 3;
    int filter_height = 3;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter is evaluated

    Convolutional *conv1_1  = new Convolutional("conv1_1", num_filters_1,
                                                 filter_width, filter_height, pad,
                                                 stride, d_layer);
    layer_list.push_back(conv1_1);

    ReLU *relu1_1 = new ReLU("relu1_1", conv1_1);
    layer_list.push_back(relu1_1);

    Convolutional *conv1_2  = new Convolutional("conv1_2", num_filters_1,
                                                 filter_width, filter_height, pad,
                                                 stride, relu1_1);
    layer_list.push_back(conv1_2);

    ReLU *relu1_2 = new ReLU("relu1_2", conv1_2);
    layer_list.push_back(relu1_2);

    int p_w = 2; // pooling width
    int p_h = 2; // height
    int p_stride = 2; // stride
    int p_pad = 0; // padding

    MaxPooling *pool1 = new MaxPooling("pool1", p_w, p_h, p_stride, p_pad, relu1_2);
    layer_list.push_back(pool1);

    int num_filters_2 = 128;
    Convolutional *conv2_1  = new Convolutional("conv2_1", num_filters_2,
                                                 filter_width, filter_height, pad,
                                                 stride, pool1);
    layer_list.push_back(conv2_1);

    ReLU *relu2_1 = new ReLU("relu2_1", conv2_1);
    layer_list.push_back(relu2_1);

    Convolutional *conv2_2  = new Convolutional("conv2_2", num_filters_2,
                                                 filter_width, filter_height, pad,
                                                 stride, relu2_1);
    layer_list.push_back(conv2_2);

    ReLU *relu2_2 = new ReLU("relu2_2", conv2_2);
    layer_list.push_back(relu2_2);

    MaxPooling *pool2 = new MaxPooling("pool2", p_w, p_h, p_stride, p_pad, relu2_2);
    layer_list.push_back(pool2);

    int num_filters_3 = 256;
    Convolutional *conv3_1  = new Convolutional("conv3_1", num_filters_3,
                                                 filter_width, filter_height, pad,
                                                 stride, pool2);
    layer_list.push_back(conv3_1);

    ReLU *relu3_1 = new ReLU("relu3_1", conv3_1);
    layer_list.push_back(relu3_1);

    Convolutional *conv3_2  = new Convolutional("conv3_2", num_filters_3,
                                                 filter_width, filter_height, pad,
                                                 stride, relu3_1);
    layer_list.push_back(conv3_2);

    ReLU *relu3_2 = new ReLU("relu3_2", conv3_2);
    layer_list.push_back(relu3_2);

    Convolutional *conv3_3  = new Convolutional("conv3_3", num_filters_3,
                                                 filter_width, filter_height, pad,
                                                 stride, relu3_2);
    layer_list.push_back(conv3_3);

    ReLU *relu3_3 = new ReLU("relu3_3", conv3_3);
    layer_list.push_back(relu3_3);

    MaxPooling *pool3 = new MaxPooling("pool3", p_w, p_h, p_stride, p_pad, relu3_3);
    layer_list.push_back(pool3);

    int num_filters_4 = 512;
    Convolutional *conv4_1  = new Convolutional("conv4_1", num_filters_4,
                                                filter_width, filter_height, pad,
                                                stride, pool3);
    layer_list.push_back(conv4_1);

    ReLU * relu4_1 = new ReLU("relu4_1", conv4_1);
    layer_list.push_back(relu4_1);

    Convolutional *conv4_2  = new Convolutional("conv4_2", num_filters_4,
                                                filter_width, filter_height, pad,
                                                stride, relu4_1);
    layer_list.push_back(conv4_2);

    ReLU *relu4_2 = new ReLU("relu4_2", conv4_2);
    layer_list.push_back(relu4_2);

    Convolutional *conv4_3  = new Convolutional("conv4_3", num_filters_4,
                                                 filter_width, filter_height, pad,
                                                stride, relu4_2);
    layer_list.push_back(conv4_3);

    ReLU *relu4_3 = new ReLU("relu4_3", conv4_3);
    layer_list.push_back(relu4_3);

    MaxPooling *pool4 = new MaxPooling("pool4", p_w, p_h, p_stride, p_pad, relu4_3);
    layer_list.push_back(pool4);

    int num_filters_5 = 512;
    Convolutional *conv5_1  = new Convolutional("conv5_1", num_filters_5,
                                                filter_width, filter_height, pad,
                                                stride, pool4);
    layer_list.push_back(conv5_1);

    ReLU *relu5_1 = new ReLU("relu5_1", conv5_1);
    layer_list.push_back(relu5_1);

    Convolutional *conv5_2  = new Convolutional("conv5_2", num_filters_5,
                                                 filter_width, filter_height, pad,
                                                 stride, relu5_1);
    layer_list.push_back(conv5_2);

    ReLU *relu5_2 = new ReLU("relu5_2", conv5_2);
    layer_list.push_back(relu5_2);

    Convolutional *conv5_3  = new Convolutional("conv5_3", num_filters_5,
                                                 filter_width, filter_height, pad,
                                                 stride, relu5_2);
    layer_list.push_back(conv5_3);

    ReLU *relu5_3 = new ReLU("relu5_3", conv5_3);
    layer_list.push_back(relu5_3);

    MaxPooling *pool5 = new MaxPooling("pool5", p_w, p_h, p_stride, p_pad, relu5_3);
    layer_list.push_back(pool5);

    Flatten *flatten = new Flatten("flatten", pool5);
    layer_list.push_back(flatten);

    int fc6_out_dim = 4096;

    Affine *fc6 = new Affine("fc6", fc6_out_dim, flatten);
    layer_list.push_back(fc6);

    ReLU *relu6 = new ReLU("relu6", fc6);
    layer_list.push_back(relu6);

    // TODO: add drop out for completeness. dropout is a passthrough
    // in the forward pass.

    int fc7_out_dim = 4096;

    Affine *fc7 = new Affine("fc7", fc7_out_dim, relu6);
    layer_list.push_back(fc7);

    ReLU *relu7 = new ReLU("relu7", fc7);
    layer_list.push_back(relu7);

    // TODO: add drop out for completeness. dropout is a passthrough
    // in the forward pass.

    int num_classes = 1000;
    Affine *fc8 = new Affine("fc8", num_classes, relu7);
    layer_list.push_back(fc8);

    SoftMax *softm = new SoftMax("prob", fc8);
    layer_list.push_back(softm);

    sub_pipeline_end_points.push_back(std::make_pair(softm->name, "input"));

    // Add the layers to the layer map
    for (auto l: layer_list) {
        layers[l->name] = l;
    }
}

class GoogleNet: public Network {
    public:
        GoogleNet() : Network() {}
        void define_forward(int batch_size, int data_width, int data_height);
};

// the function below is a subroutine used by the
// GoogleNet::define_forward() method when constructing the Inception
// topology
Layer* inception_module(Layer *in, std::vector<Layer*> &layer_list, std::string prefix,
                        int _1x1_filters, int _3x3_reduce_filters, int _3x3_filters,
                        int _5x5_reduce_filters, int _5x5_filters,
                        int _pool_proj_filters) {
    Convolutional *_1x1;
    {
    int num_filters = _1x1_filters;
    int filter_width = 1;
    int filter_height = 1;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter is evaluated

    _1x1  = new Convolutional(prefix + "1x1", num_filters,
                              filter_width, filter_height, pad, stride, in);
    layer_list.push_back(_1x1);
    }

    ReLU *relu_1x1 = new ReLU(prefix + "relu_1x1", _1x1);
    layer_list.push_back(relu_1x1);

    Convolutional *_3x3_reduce;
    {
    int num_filters = _3x3_reduce_filters;
    int filter_width = 1;
    int filter_height = 1;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter is evaluated

    _3x3_reduce  = new Convolutional(prefix + "3x3_reduce", num_filters,
                                     filter_width, filter_height, pad, stride, in);
    layer_list.push_back(_3x3_reduce);
    }

    ReLU *relu_3x3_reduce = new ReLU(prefix + "relu_3x3_reduce", _3x3_reduce);
    layer_list.push_back(relu_3x3_reduce);

    Convolutional *_3x3;
    {
    int num_filters = _3x3_filters;
    int filter_width = 3;
    int filter_height = 3;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter is evaluated

    _3x3  = new Convolutional(prefix + "3x3", num_filters,
                              filter_width, filter_height, pad,
                              stride, relu_3x3_reduce);
    layer_list.push_back(_3x3);
    }

    ReLU *relu_3x3 = new ReLU(prefix + "relu_3x3", _3x3);
    layer_list.push_back(relu_3x3);

    Convolutional *_5x5_reduce;
    {
    int num_filters = _5x5_reduce_filters;
    int filter_width = 1;
    int filter_height = 1;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter is evaluated

    _5x5_reduce  = new Convolutional(prefix + "5x5_reduce", num_filters,
                                     filter_width, filter_height, pad,
                                     stride, in);
    layer_list.push_back(_5x5_reduce);
    }

    ReLU *relu_5x5_reduce = new ReLU(prefix + "relu_5x5_reduce", _5x5_reduce);
    layer_list.push_back(relu_5x5_reduce);

    Convolutional *_5x5;
    {
    int num_filters = _5x5_filters;
    int filter_width = 5;
    int filter_height = 5;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter is evaluated

    _5x5  = new Convolutional(prefix + "5x5", num_filters,
                              filter_width, filter_height, pad,
                              stride, relu_5x5_reduce);
    layer_list.push_back(_5x5);
    }

    ReLU *relu_5x5 = new ReLU(prefix + "relu_5x5", _5x5);
    layer_list.push_back(relu_5x5);

    int p_w = 3; // pooling width
    int p_h = 3; // height
    int p_stride = 1; // stride
    int p_pad = 1; // padding

    MaxPooling *pool = new MaxPooling(prefix + "pool", p_w, p_h,
                                      p_stride, p_pad, in);
    layer_list.push_back(pool);

    Convolutional *pool_proj;
    {
    int num_filters = _pool_proj_filters;
    int filter_width = 1;
    int filter_height = 1;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter is evaluated

    pool_proj  = new Convolutional(prefix + "pool_proj", num_filters,
                                   filter_width, filter_height, pad, stride, pool);
    layer_list.push_back(pool_proj);
    }

    ReLU *relu_pool_proj = new ReLU(prefix + "relu_pool_proj", pool_proj);
    layer_list.push_back(relu_pool_proj);

    std::vector<Layer*> concat_inputs;
    concat_inputs.push_back(relu_1x1);
    concat_inputs.push_back(relu_3x3);
    concat_inputs.push_back(relu_5x5);
    concat_inputs.push_back(relu_pool_proj);
    Concat *output = new Concat(prefix + "output", concat_inputs);
    layer_list.push_back(output);

    return output;
}

/*
 * GoogleNet::define_forward --
 *
 * This function builds the topology of the InceptionV1 network by
 * instantiating the appropriate "Layer" classes and writing up their
 * inputs and outputs.
 */
void GoogleNet::define_forward(int batch_size, int data_width, int data_height) {
    std::vector<Layer*> layer_list;

    // Halide cannot compile googlenet if it is done as a single
    // pipeline (the DAG is too big), so it is built as multiple
    // Halide pipelines chained together.
    int channels = 3;

    Image<float> input(data_width, data_height, channels, batch_size);

    DataLayer *d_layer_0 = new DataLayer("input", input);
    layer_list.push_back(d_layer_0);

    Convolutional *conv1_7x7_s2;
    {
    int num_filters = 64;
    int filter_width = 7;
    int filter_height = 7;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 2; // stride at which the filter is evaluated

    conv1_7x7_s2  = new Convolutional("conv1/7x7_s2", num_filters,
                                       filter_width, filter_height, pad,
                                       stride, d_layer_0);
    layer_list.push_back(conv1_7x7_s2);
    }

    ReLU *conv1_relu_7x7 = new ReLU("conv1/relu_7x7", conv1_7x7_s2);
    layer_list.push_back(conv1_relu_7x7);

    int p_w = 3; // pooling width
    int p_h = 3; // height
    int p_stride = 2; // stride
    int p_pad = 0; // padding

    MaxPooling *pool1_3x3_s2 = new MaxPooling("pool1/3x3_s2", p_w, p_h,
                                               p_stride, p_pad, conv1_relu_7x7);
    layer_list.push_back(pool1_3x3_s2);

    int local_size = 5;
    float alpha = 0.0001;
    float beta = 0.75;
    LRN *pool1_norm1  = new LRN("pool1/norm1", local_size, alpha, beta, pool1_3x3_s2);

    layer_list.push_back(pool1_norm1);

    Convolutional *conv2_3x3_reduce;
    {
    int num_filters = 64;
    int filter_width = 1;
    int filter_height = 1;
    int pad = (filter_height - 1)/2;
    int stride = 1;
    conv2_3x3_reduce  = new Convolutional("conv2/3x3_reduce", num_filters, filter_width,
                                           filter_height, pad, stride, pool1_norm1);
    layer_list.push_back(conv2_3x3_reduce);
    }

    ReLU *relu_3x3_reduce = new ReLU("conv2/relu_3x3_reduce", conv2_3x3_reduce);
    layer_list.push_back(relu_3x3_reduce);

    Convolutional *_3x3;
    {
    int num_filters = 192;
    int filter_width = 3;
    int filter_height = 3;
    int pad = (filter_height - 1)/2;
    int stride = 1;
    _3x3  = new Convolutional("conv2/3x3", num_filters, filter_width,
                               filter_height, pad, stride, relu_3x3_reduce);
    layer_list.push_back(_3x3);
    }

    ReLU *relu_3x3 = new ReLU("conv2/relu_3x3", _3x3);
    layer_list.push_back(relu_3x3);

    LRN *conv2_norm2  = new LRN("conv2/norm2", local_size, alpha, beta, relu_3x3);

    layer_list.push_back(conv2_norm2);

    MaxPooling *pool2_3x3_s2 = new MaxPooling("pool2/3x3_s2", p_w, p_h,
                                               p_stride, p_pad, conv2_norm2);
    layer_list.push_back(pool2_3x3_s2);

    sub_pipeline_end_points.push_back(std::make_pair(pool2_3x3_s2->name, "input"));

    Image<float> inter_1(pool2_3x3_s2->out_dim_size(0), pool2_3x3_s2->out_dim_size(1),
                         pool2_3x3_s2->out_dim_size(2), pool2_3x3_s2->out_dim_size(3));

    DataLayer *d_layer_1 = new DataLayer("inter_1", inter_1);
    layer_list.push_back(d_layer_1);

    Layer *output_3a = inception_module(d_layer_1, layer_list, "inception_3a/",
                                        64, 96, 128, 16, 32, 32);
    Layer *output_3b = inception_module(output_3a, layer_list, "inception_3b/",
                                        128, 128, 192, 32, 96, 64);

    MaxPooling *pool3_3x3_s2 = new MaxPooling("pool3/3x3_s2", p_w, p_h,
                                               p_stride, p_pad, output_3b);
    layer_list.push_back(pool3_3x3_s2);

    sub_pipeline_end_points.push_back(std::make_pair(pool3_3x3_s2->name, "inter_1"));

    Image<float> inter_2(pool3_3x3_s2->out_dim_size(0), pool3_3x3_s2->out_dim_size(1),
                         pool3_3x3_s2->out_dim_size(2), pool3_3x3_s2->out_dim_size(3));

    DataLayer *d_layer_2 = new DataLayer("inter_2", inter_2);
    layer_list.push_back(d_layer_2);

    Layer *output_4a = inception_module(d_layer_2, layer_list, "inception_4a/",
                                        192, 96, 208, 16, 48, 64);
    Layer *output_4b = inception_module(output_4a, layer_list, "inception_4b/",
                                        160, 112, 224, 24, 64, 64);

    sub_pipeline_end_points.push_back(std::make_pair(output_4b->name, "inter_2"));

    Image<float> inter_2_1(output_4b->out_dim_size(0), output_4b->out_dim_size(1),
                           output_4b->out_dim_size(2), output_4b->out_dim_size(3));

    DataLayer *d_layer_2_1 = new DataLayer("inter_2_1", inter_2_1);
    layer_list.push_back(d_layer_2_1);

    Layer *output_4c = inception_module(d_layer_2_1, layer_list, "inception_4c/",
                                        128, 128, 256, 24, 64, 64);
    Layer *output_4d = inception_module(output_4c, layer_list, "inception_4d/",
                                        112, 144, 288, 32, 64, 64);

    sub_pipeline_end_points.push_back(std::make_pair(output_4d->name, "inter_2_1"));

    Image<float> inter_2_2(output_4d->out_dim_size(0), output_4d->out_dim_size(1),
                           output_4d->out_dim_size(2), output_4d->out_dim_size(3));

    DataLayer *d_layer_2_2 = new DataLayer("inter_2_2", inter_2_2);

    layer_list.push_back(d_layer_2_2);
    Layer *output_4e = inception_module(d_layer_2_2, layer_list, "inception_4e/",
                                        256, 160, 320, 32, 128, 128);

    MaxPooling *pool4_3x3_s2 = new MaxPooling("pool4/3x3_s2", p_w, p_h,
                                               p_stride, p_pad, output_4e);
    layer_list.push_back(pool4_3x3_s2);

    sub_pipeline_end_points.push_back(std::make_pair(pool4_3x3_s2->name, "inter_2_2"));

    Image<float> inter_3(pool4_3x3_s2->out_dim_size(0), pool4_3x3_s2->out_dim_size(1),
                         pool4_3x3_s2->out_dim_size(2), pool4_3x3_s2->out_dim_size(3));

    DataLayer *d_layer_3 = new DataLayer("inter_3", inter_3);
    layer_list.push_back(d_layer_3);

    Layer *output_5a = inception_module(d_layer_3, layer_list, "inception_5a/",
                                        256, 160, 320, 32, 128, 128);
    Layer *output_5b = inception_module(output_5a, layer_list, "inception_5b/",
                                        384, 192, 384, 48, 128, 128);

    AvgPooling *pool5_7x7_s1;
    {

    int p_w = 7;
    int p_h = 7;
    int p_stride = 1;
    int p_pad = 0;
    pool5_7x7_s1 = new AvgPooling("pool5/7x7_s1", p_w, p_h, p_stride, p_pad, output_5b);

    }
    layer_list.push_back(pool5_7x7_s1);

    Flatten *flatten = new Flatten("flatten", pool5_7x7_s1);
    layer_list.push_back(flatten);

    int num_classes = 1000;
    Affine *loss3_classifier = new Affine("loss3/classifier", num_classes, flatten);
    layer_list.push_back(loss3_classifier);

    SoftMax *softm = new SoftMax("prob", loss3_classifier);
    layer_list.push_back(softm);

    sub_pipeline_end_points.push_back(std::make_pair(softm->name, "inter_3"));

    // Add the layers to the layer map
    for (auto l: layer_list) {
        layers[l->name] = l;
    }
}

class ToyNet: public Network {
    public:
        ToyNet() : Network() {}
        void define_forward(int batch_size, int data_width, int data_height);
        void define_backward(Image<int> labels);
};


/*
 * ToyNet::define_forward --
 *
 * This function builds the topology of the ToyNet network by
 * instantiating the appropriate "Layer" classes and writing up their
 * inputs and outputs. In this assignment you will train ToyNet to
 * perform image classification on the CIFAR 100.
 */
void ToyNet::define_forward(int batch_size, int data_width, int data_height) {

    // Network structure
    // input -> conv1 -> relu1 -> conv2 -> relu2 -> pool ->
    // flatten -> fc -> prob

    std::vector<Layer*> layer_list;

    int channels = 3;

    Image<float> input(data_width, data_height, channels, batch_size);

    DataLayer *d_layer = new DataLayer("input", input);
    layer_list.push_back(d_layer);

    int num_filters = 32;
    int filter_width = 3;
    int filter_height = 3;
    int pad = (filter_width-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter is evaluated

    Convolutional *conv1  = new Convolutional("conv1", num_filters,
                                               filter_width, filter_height, pad,
                                               stride, d_layer);
    layer_list.push_back(conv1);

    ReLU *relu1 = new ReLU("relu1", conv1);
    layer_list.push_back(relu1);

    Convolutional *conv2  = new Convolutional("conv2", num_filters,
                                               filter_width, filter_height, pad,
                                               stride, relu1);
    layer_list.push_back(conv2);

    ReLU *relu2 = new ReLU("relu2", conv2);
    layer_list.push_back(relu2);

    int p_w = 2; // pooling width
    int p_h = 2; // height
    int p_stride = 2; // stride
    int p_pad = 0; // padding

    MaxPooling *pool1 = new MaxPooling("pool1", p_w, p_h, p_stride, p_pad, relu2);
    layer_list.push_back(pool1);

    Flatten *flatten = new Flatten("flatten", pool1);
    layer_list.push_back(flatten);

    int num_classes = 100;
    int fc_out_dim = num_classes;

    Affine *fc = new Affine("fc", fc_out_dim, flatten);
    layer_list.push_back(fc);

    SoftMax *softm = new SoftMax("prob", fc);
    layer_list.push_back(softm);

    sub_pipeline_end_points.push_back(std::make_pair(softm->name, "input"));

    // Add the layers to the layer map
    for (auto l: layer_list) {
        layers[l->name] = l;
    }
}

void ToyNet::define_backward(Image<int> labels) {
    layers["prob"]->define_gradients(Func(labels));
    layers["fc"]->define_gradients(layers["prob"]->f_input_grads[0]);
    layers["flatten"]->define_gradients(layers["fc"]->f_input_grads[0]);
    layers["pool1"]->define_gradients(layers["flatten"]->f_input_grads[0]);
    layers["relu2"]->define_gradients(layers["pool1"]->f_input_grads[0]);
    layers["conv2"]->define_gradients(layers["relu2"]->f_input_grads[0]);
    layers["relu1"]->define_gradients(layers["conv2"]->f_input_grads[0]);
    layers["conv1"]->define_gradients(layers["relu1"]->f_input_grads[0]);
}

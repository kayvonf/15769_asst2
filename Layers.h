#include<iostream>
#include "Halide.h"

using namespace Halide;

// sanity check to make sure the halide function is defined.
void check_defined(Func f) {
    if (!f.defined()) {
        std::cout << f.name() << " is undefined" << std::endl;
        exit(-1);
    }
}

/*
 * Layer --
 *
 * Base class for a layer.  Students will likely not need to change
 * any code in this layer.  However, in this assignment students
 * *will* need to implement subclasses such as Affine, Convolutional,
 * MaxPool, etc.
 */
class Layer {
    public:
        Layer(std::string _name, Layer* in) : name(_name) {
            if (in) {
                check_defined(in->forward);
                inputs.push_back(in);
                in->outputs.push_back(this);
            }
            Func _forward(name + "_forward");
            forward = _forward;
        }

       // Constructor for a layer with multiple inputs
        Layer(std::string _name, std::vector<Layer*> &in) : name(_name) {
            for (size_t i = 0; i < in.size(); i++) {
                check_defined(in[i]->forward);
                inputs.push_back(in[i]);
                in[i]->outputs.push_back(this);
            }
            Func _forward(name + "_forward");
            forward = _forward;
        }

       virtual ~Layer() {};

        // layer name
        std::string name;

        // layers which are inputs to the layer
        std::vector<Layer*> inputs;

        // layers which use the output of this layer
        std::vector<Layer*> outputs;

        // number of output dimensions of layer output
        //
        // Example: conv layers for networks that process batches of
        // images will have out_dims() = 4, with the following
        // dimensions:
        //   -- x,y
        //   -- filters
        //   -- images (in a batch)
        virtual int out_dims() = 0;

        // size of output dimension dim, 0 <= dim < out_dims(out_idx)
        //
        // Example: a layer with an output activation that is
        // 32x64x128 will have out_dim_size(0) = 32
        virtual int out_dim_size(int dim) = 0;

        // storage for layer parameters and gradients.  params_cache
        // is a copy of older params values needed for training via
        // gradient descent with momentum
        std::vector<Image<float>> params;
        std::vector<Image<float>> param_grads;
        std::vector<Image<float>> params_cache;

        // function that computes the output of the layer.
        Func forward;

        // functions which compute the gradients with respect to layer
        // parameters. e.g., there will be seperate functions for
        // dL/dW and dL/db in a layer with weight matrix W and bias
        // vector b.
        std::vector<Func> f_param_grads;

        // function which compute gradients with respect to layer inputs
        std::vector<Func> f_input_grads;

        // defines the functions which compute gradient of the
        // objective function with respect to the parameters and
        // input. f_param_grads and f_input_grads will be populated if
        // this function is called.
        //
        // dout is a function which computes the derivative of the
        // objective with respect to layer
        // output. (dLoss/dLayerOutput).
        //
        // Recall: dLoss/dW = dLoss/dLayerOutput * dLayerOutput/dW
        //
        // So the param gradient functions depend on the values of dout
        virtual void define_gradients(Func dout, bool schedule = true) = 0;
};

/*
 * Affine layer definition:
 *
 * Given dense weight parameter matrix W.
 * Bias vector b
 * and input vector x
 *
 * Implements a fully-connected layer: out = Wx + b
 */
class Affine: public Layer {
    public:

        // num_units is the number of units in the layer.  As this is
        // a fully connected layer, each unit is connect to each
        // unit in the input layer.
        //
        // num_inputs is the size (number of units) of each input
        // sample (a "sample" would correspond to an image in a batch)
        //
        // So when running a network with image batch size = 10, and
        // if the input size of the affine layer is 128 and the output
        // size is 32, then:
        //
        // num_units = 32
        // num_samples = 10
        // num_inputs = 128
        //
        // and the W matrix will be of size 128x32 and the b vector is
        // of size 32
        int num_units, num_inputs, num_samples;

        // Halide variables
        Var in_dim, unit_dim, n;

         Affine(std::string _name, int _num_units, Layer* in,
               bool schedule = true) : Layer(_name, in) {

            Func in_f = inputs[0]->forward;
            num_units = _num_units;

            // create parameters
            num_inputs = inputs[0]->out_dim_size(0);
            num_samples = inputs[0]->out_dim_size(1);

            Image<float> W(num_inputs, num_units), b(num_units);
            params.push_back(W);
            params.push_back(b);

            ////////////////////////////////////////////////////////////////////
            // start student code here
            //
            // The code should define forward(unit_dim, n) = ...
            ////////////////////////////////////////////////////////////////////

            forward(unit_dim, n) = 0;

            if (schedule) {
                // put schedule here (if scheduling layers independently)
                forward.compute_root();
            }

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////
        }

        void define_gradients(Func dout, bool schedule = true) {

            check_defined(dout);
            assert(f_input_grads.size() == 0);

            Func in_grad(name + "_in_grad");
            Func dW(name + "_dW"), db(name + "_db");

            Image<float> W = params[0];
            Image<float> b = params[1];

            // create storage for gradients and caching params
            Image<float> W_grad(num_inputs, num_units);
            param_grads.push_back(W_grad);
            Image<float> W_cache(num_inputs, num_units);
            params_cache.push_back(W_cache);

            Image<float> b_grad(num_units);
            param_grads.push_back(b_grad);
            Image<float> b_cache(num_units);
            params_cache.push_back(b_cache);

            ////////////////////////////////////////////////////////////////////
            // begin student code here
            //
            // Code should define dW(in_dim, unit_dim) ...
            //                    dB(unit_dim) ...
            //                    in_grad(in_dim, n) ...
            ////////////////////////////////////////////////////////////////////

            in_grad(in_dim, n) = 0.f;
            dW(in_dim, unit_dim) = 0.f;
            db(unit_dim) = 0.f;

            if (schedule) {
                // put schedule here (if scheduling layers independently)
                dW.compute_root();
                db.compute_root();
                in_grad.compute_root();
            }

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////

            f_param_grads.push_back(dW);
            f_param_grads.push_back(db);

            f_input_grads.push_back(in_grad);
        }

        int out_dims() { return 2; }

        int out_dim_size(int i) {
            assert(i < 2);
            int size = 0;
            if(i == 0) {
                size = num_units;
            } else if(i == 1) {
                size = num_samples;
            }
            return size;
        }
};

/*
 * Convolutional Layer --
 *
 */
class Convolutional: public Layer {
    public:

        // number of channels, height and width of the input to the layer
        //
        // Example: if the input is a 224x224 RGB images, and the
        // image batch size = 10 then:
        //   -- num_samples = 10
        //   -- in_ch = 3
        //   -- in_h = in_w = 224
        int num_samples, in_ch, in_h, in_w;

        // number of filters in the convolution layer, filter height,
        // filter width, padding and stride
        int num_f, f_h, f_w, pad, stride;

        // Halide vars
        Var x, y, z, n;

        // padded input to avoid bounds check during the computation
        Func f_in_bound;

        Convolutional(std::string _name, int _num_f, int _f_w, int _f_h,
                      int _pad, int _stride, Layer* in,
                      bool schedule=true) : Layer(_name, in) {

            Func _f_in_bound(name + "_f_in_bound");
            f_in_bound = _f_in_bound;

            assert(inputs[0]->out_dims() == 4);

            // input layout: width, height, channels, samples
            num_samples = inputs[0]->out_dim_size(3);
            in_ch = inputs[0]->out_dim_size(2);
            in_h = inputs[0]->out_dim_size(1);
            in_w = inputs[0]->out_dim_size(0);

            num_f = _num_f;
            f_h = _f_h;
            f_w = _f_w;
            pad = _pad;
            stride = _stride;

            // create a padded input and avoid checking boundary
            // conditions while computing the actual convolution
            f_in_bound = BoundaryConditions::constant_exterior(inputs[0]->forward, 0,
                                                               0, in_w, 0, in_h);

            // create parameters
            Image<float> W(f_w, f_h, in_ch, num_f), b(num_f);
            params.push_back(W);
            params.push_back(b);

            ////////////////////////////////////////////////////////////////////
            // start student code here
            //
            // Code should define forward(x, y, z, n) = ...
            ////////////////////////////////////////////////////////////////////

            forward(x, y, z, n) = 0.f;

            if (schedule) {
                // put schedule here (if scheduling layers independently)
                f_in_bound.compute_root();
                forward.compute_root();
            }

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////

        }

       /*
        * define_gradients --
        *
        * Generates functions to compute layer parameter and layer input
        * gradients given dout = dLoss/layer
        */
        void define_gradients(Func dout, bool schedule = true) {
            check_defined(dout);
            assert(f_input_grads.size() == 0);

            Func in_grad(name + "_in_grad");
            Func dW(name + "_dW"), db(name + "_db");

            //int out_w = this->out_dim_size(0);
            //int out_h = this->out_dim_size(1);

            Image<float> W = params[0];
            Image<float> b = params[1];

            // create storage for gradients and caching params
            Image<float> W_grad(f_w, f_h, in_ch, num_f);
            param_grads.push_back(W_grad);
            Image<float> W_cache(f_w, f_h, in_ch, num_f);
            params_cache.push_back(W_cache);

            Image<float> b_grad(num_f);
            param_grads.push_back(b_grad);
            Image<float> b_cache(num_f);
            params_cache.push_back(b_cache);

            ////////////////////////////////////////////////////////////////////
            // start student code here
            //
            // Code should define dW(x, y, z, n) = ...
            //                    db(x) = ...
            //                    in_grad(x, y, z, n) = ...
            ////////////////////////////////////////////////////////////////////


            dW(x, y, z, n) = 0.f;
            db(x) = 0.f;
            in_grad(x, y, z, n) = 0.f;

            if (schedule) {
                // put schedule here (if scheduling layers independently)
                dW.compute_root();
                db.compute_root();
                in_grad.compute_root();
            }

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////

            f_param_grads.push_back(dW);
            f_param_grads.push_back(db);

            f_input_grads.push_back(in_grad);
        }

        int out_dims() { return 4; }

        int out_dim_size(int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0) {
                size = (1 + (in_w + 2 * pad - f_w)/stride);
            } else if (i == 1) {
                size = (1 + (in_h + 2 * pad - f_h)/stride);
            } else if (i == 2) {
                size = num_f;
            } else if (i == 3) {
                size = num_samples;
            }
            return size;
        }
};

class MaxPooling: public Layer {
    public:

        // number of color channels in input in_c
        // height and width of the input in_h, in_w
        int num_samples, in_ch, in_h, in_w;

        // height and width of the pool
        // stride at which the pooling is applied
        int p_h, p_w, stride, pad;

        // Halide variables
        Var x, y, z, n;

        // padded input to avoid need to check boundary conditions
        Func f_in_bound;

        MaxPooling(std::string _name, int _p_w, int _p_h, int _stride,
                   int _pad, Layer* in, bool schedule = true) : Layer(_name, in) {
            assert(inputs[0]->out_dims() == 4);

            num_samples = inputs[0]->out_dim_size(3);
            in_ch = inputs[0]->out_dim_size(2);
            in_h = inputs[0]->out_dim_size(1);
            in_w = inputs[0]->out_dim_size(0);

            p_w = _p_w;
            p_h = _p_h;
            stride = _stride;
            pad = _pad;

            Func in_f = inputs[0]->forward;

            // create a padded input and avoid checking boundary
            // conditions while computing the max in the pool widow
            f_in_bound = BoundaryConditions::constant_exterior(in_f, 0,
                                                               0, in_w, 0, in_h);

            ////////////////////////////////////////////////////////////////////
            // begin student code here
            //
            // the code should define: forward(x, y, z, n) = ...
            ////////////////////////////////////////////////////////////////////

            forward(x, y, z, n) = 0.f;

            if (schedule) {
                // put schedule here (if scheduling layers independently)
                f_in_bound.compute_root();
                forward.compute_root();
            }

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////
        }

       /*
        * define_gradients --
        *
        * Generates functions to compute layer parameter and layer input
        * gradients given dout = dLoss/layer
        */
        void define_gradients(Func dout, bool schedule = true) {
            check_defined(dout);
            assert(f_input_grads.size() == 0);

            Func in_f = inputs[0]->forward;

            Func in_grad(name + "_in_grad");

            ////////////////////////////////////////////////////////////////////
            // begin student code here
            //
            // The code should define in_grad() ...
            ////////////////////////////////////////////////////////////////////

            in_grad(x, y, z, n) = 0.f;

            if (schedule) {
                // put schedule here (if scheduling layers independently)
                in_grad.compute_root();
            }

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////

            f_input_grads.push_back(in_grad);

        }

        int out_dims() { return 4; }

        int out_dim_size(int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0) {
                // Matching caffe's weird behavior
                size = 1 + std::ceil((float)(in_w + 2 * pad - p_w)/stride);
            } else if (i == 1) {
                // Matching caffe's weird behavior
                size = 1 + std::ceil((float)(in_h + 2 * pad - p_h)/stride);
            } else if (i == 2) {
                size = inputs[0]->out_dim_size(2);
            } else if (i == 3) {
                size = num_samples;
            }
            return size;
        }
};

/*
 * ReLU layer --
 */
class ReLU: public Layer {
    public:
        Var x, y, z, w;
        ReLU(std::string _name, Layer* in, bool schedule = true) : Layer(_name, in) {
            Func in_f = inputs[0]->forward;

            // define forward
            switch(inputs[0]->out_dims()) {

                case 1:
                    forward(x) = max(0, in_f(x));
                   break;
                case 2:
                    forward(x, y) = max(0, in_f(x, y));
                    break;
                case 3:
                    forward(x, y, z) = max(0, in_f(x, y, z));
                    break;
                case 4:
                    forward(x, y, z, w) = max(0, in_f(x, y, z, w));
                    break;
                default:
                    std::cout << "ReLU layer does not support inputs with more\
                                  than 4 dimensions" << std::endl;
            }
        }

        void define_gradients(Func dout, bool schedule = true) {
            check_defined(dout);
            assert(f_input_grads.size() == 0);

            Func in_f = inputs[0]->forward;
            Func in_grad(name + "_in_grad");
            switch(inputs[0]->out_dims()) {
                case 1:
                    in_grad(x) = dout(x) * select(in_f(x) > 0, 1, 0);
                    break;
                case 2:
                    in_grad(x, y) = dout(x, y) * select(in_f(x, y) > 0, 1, 0);
                    break;
                case 3:
                    in_grad(x, y, z) = dout(x, y, z) * select(in_f(x, y, z) > 0, 1, 0);
                    break;
                case 4:
                    in_grad(x, y, z, w) = dout(x, y, z, w) * select(in_f(x, y, z, w) > 0, 1, 0);
                    break;
                default:
                    assert(0);
            }
            f_input_grads.push_back(in_grad);

            if (schedule) {
                in_grad.compute_root();
            }
        }

        int out_dims() { return inputs[0]->out_dims(); }

        int out_dim_size(int i) {
            return inputs[0]->out_dim_size(i);
        }
};

/*
 * Softmax layer -- converts scores for num_classes into [0-1] values.
 *
 */
class SoftMax: public Layer {
    public:

        // Expects 2-dimensional input layer (num_classes x num_samples)
        int num_classes, num_samples;

        // Halide vars
        Var in_dim, n;

        SoftMax(std::string _name, Layer* in,
                bool schedule = true) : Layer(_name, in) {

            assert(in->out_dims() == 2);
            Func in_f = inputs[0]->forward;

            num_classes = in->out_dim_size(0);
            num_samples = in->out_dim_size(1);

            ////////////////////////////////////////////////////////////////////
            // begin student code here
            //
            // Code should define forward(in_dim, n) = ...
            ////////////////////////////////////////////////////////////////////

            forward(in_dim, n) = 0.f;

            if (schedule) {
                // put schedule here (if scheduling layers independently)
                forward.compute_root();
            }

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////
        }

        /*
        * define_gradients --
        *
        * Generates functions to compute layer parameter and layer input
        * gradients given dout = dLoss/layer
        */
        void define_gradients(Func labels, bool schedule = true) {
            check_defined(labels);

            Func in_grad(name + "_in_grad");

            ////////////////////////////////////////////////////////////////////
            // begin student code here
            //
            // Code should define in_grad(in_dim, n) = ...
            //
            // Note there are no parameter derivatives since the
            // softMax has no learnable parameters.
            ////////////////////////////////////////////////////////////////////

            in_grad(in_dim, n) = 0.f;

            if (schedule) {
                // put schedule here (if scheduling layers independently)
                in_grad.compute_root();
            }

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////

            f_input_grads.push_back(in_grad);
        }

        // Returns a halide function that computes softmax loss given
        // the correct labels for each sample
        Func loss(Func labels) {

            // Check if labels is defined
            check_defined(labels);
            // Check if the dimensions make sense
            assert(labels.dimensions() == 1);

            Var x;
            Func loss_p(name + "_loss_p");

            ////////////////////////////////////////////////////////////////////
            // begin student code here
            //
            // Code should define loss_p(x) = ...
            ////////////////////////////////////////////////////////////////////

            loss_p(x) = 0.f;

            ////////////////////////////////////////////////////////////////////
            // end student code here
            ////////////////////////////////////////////////////////////////////

            return loss_p;
        }

        int out_dims() { return 2; }

        int out_dim_size(int i) {
            assert(i < 2);
            int size = 0;
            if (i == 0) {
                size = num_classes;
            } else if (i == 1) {
                size = num_samples;
            }
            return size;
        }
};

/*
 * A DropOut layer is used as regularization during training.  It is
 * non-existent in the network during inference.
 */
class DropOut: public Layer {
    public:
        Var x, y, z, w;

        // Threshold value between 0-1 the probability
        // with which a unit's output will be dropped
        float thresh;

        // Mask containing the drop out coefficients in the forward pass
        Func mask;

        DropOut(std::string _name, float _thresh, Layer *in,
                bool schedule = true) : Layer(_name, in) {

            Func _mask(name + "_mask");
            mask = _mask;
            thresh = _thresh;
            Func in_f = inputs[0]->forward;

            // define forward
            Expr scale = 1.0f/(1.0f - thresh);
            switch(inputs[0]->out_dims()) {
                case 1:
                    mask(x) = select(random_float(in_f(x)) > thresh, scale, 0.0f);
                    forward(x) = mask(x) * in_f(x);
                    break;
                case 2:
                    mask(x, y) = select(random_float(in_f(x, y)) > thresh, scale, 0.0f);
                    forward(x, y) = mask(x, y) * in_f(x, y);
                    break;
                case 3:
                    mask(x, y, z) = select(random_float(in_f(x, y, z)) > thresh, scale, 0.0f);
                    forward(x, y, z) = mask(x, y, z) * in_f(x, y, z);
                    break;
                case 4:
                    mask(x, y, z, w) = select(random_float(in_f(x, y, z, w)) > thresh, scale, 0.0f);
                    forward(x, y, z, w) = mask(x, y, z, w) * in_f(x, y, z, w);
                    break;
                default:
                    std::cout << "Dropout layer does not support inputs with more\
                                  than 4 dimensions" << std::endl;
            }
        }

        void define_gradients(Func dout, bool schedule = true) {
            check_defined(dout);
            assert(f_input_grads.size() == 0);

            Func in_grad(name + "_in_grad");
            switch(inputs[0]->out_dims()) {
                case 1:
                    in_grad(x) = dout(x) * mask(x);
                    break;
                case 2:
                    in_grad(x, y) = dout(x, y) * mask(x, y);
                    break;
                case 3:
                    in_grad(x, y, z) = dout(x, y, z) * mask(x, y, z);
                    break;
                case 4:
                    in_grad(x, y, z, w) = dout(x, y, z, w) * mask(x, y, z, w);
                    break;
                default:
                    assert(0);
            }
            f_input_grads.push_back(in_grad);

            if (schedule) {
                in_grad.compute_root();
            }
        }

        int out_dims() { return inputs[0]->out_dims(); }

        int out_dim_size(int i) {
            return inputs[0]->out_dim_size(i);
        }
};


/*
 * Local Response Normalization layer --
 *
 */
class LRN: public Layer {
    public:

        // number of channels, height and width of the input to the layer
        int num_samples, in_ch, in_h, in_w;

        // Response normalization window size
        int w_size;
        float alpha, beta;

        // Halide vars
        Var x, y, z, n;

        // padded input to avoid need to check boundary conditions
        Func f_in_bound;

        LRN(std::string _name, int _w_size, float _alpha, float _beta,
            Layer* in, bool schedule = true) : Layer(_name, in) {

            Func _f_in_bound(name + "_f_in_bound");
            f_in_bound = _f_in_bound;

            Func in_f = inputs[0]->forward;

            assert(inputs[0]->out_dims() == 4);

            num_samples = inputs[0]->out_dim_size(3);
            in_ch = inputs[0]->out_dim_size(2);
            in_h = inputs[0]->out_dim_size(1);
            in_w = inputs[0]->out_dim_size(0);

            w_size = _w_size;
            alpha = _alpha;
            beta = _beta;

            // create a padded input and avoid checking boundary
            // conditions while computing the actual convolution
            f_in_bound = BoundaryConditions::constant_exterior(inputs[0]->forward, 0,
                                                               0, in_w, 0, in_h, 0, in_ch);

            // define forward
            RDom r(0, w_size);

            Func square_sum(name + "_square_sum");
            square_sum(x, y, z, n) = cast(in_f.output_types()[0], 0);
            Expr val = f_in_bound(x, y, z + r.x - w_size/2, n);
            square_sum(x, y, z, n) += val * val;

            Expr norm_factor = pow(1.0f + (alpha/(w_size)) * square_sum(x, y, z, n), beta);
            forward(x, y, z, n) = f_in_bound(x, y, z, n)/norm_factor;

            if (schedule) {
                square_sum.compute_root().parallel(n).parallel(z).vectorize(x, 8);
                square_sum.update().parallel(n).parallel(z).vectorize(x, 8);
                forward.compute_root().parallel(n).parallel(z).vectorize(x, 8);
            }
        }

        void define_gradients(Func dout, bool schedule = true) {
            check_defined(dout);
            assert(f_input_grads.size() == 0);

            // Gradients for Local Response Normalization not implemented at this time.

            if (schedule) {
            }
        }

        int out_dims() { return 4; }

        int out_dim_size(int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0) {
                return in_w;
            } else if (i == 1) {
                return in_h;
            } else if (i == 2) {
                return in_ch;
            } else if (i == 3) {
                return num_samples;
            }
            return size;
        }
};

class AvgPooling: public Layer {
    public:

        // number of color channels in input in_c
        // height and width of the input in_h, in_w
        int num_samples, in_ch, in_h, in_w;

        // height and width of the pool (example: 2x2 pooling, each
        // unit is averaging four inputs) and the stride at which the
        // pooling is applied
        int p_h, p_w, stride, pad;

        // Halide vars
        Var x, y, z, n;

        // padded input to avoid need to check boundary conditions
        Func f_in_bound;

        AvgPooling(std::string _name, int _p_w, int _p_h, int _stride,
                   int _pad, Layer* in, bool schedule = true) : Layer(_name, in) {
            assert(inputs[0]->out_dims() == 4);

            num_samples = inputs[0]->out_dim_size(3);
            in_ch = inputs[0]->out_dim_size(2);
            in_h = inputs[0]->out_dim_size(1);
            in_w = inputs[0]->out_dim_size(0);

            p_w = _p_w;
            p_h = _p_h;
            stride = _stride;
            pad = _pad;

            Func in_f = inputs[0]->forward;
            // create a padded input and avoid checking boundary
            // conditions while computing the max in the pool widow
            f_in_bound = BoundaryConditions::constant_exterior(in_f, 0,
                                                               0, in_w, 0, in_h);
            // define forward
            RDom r(0, p_w, 0, p_h);
            forward(x, y, z, n) = sum(f_in_bound(x * stride + r.x - pad,
                                                 y * stride + r.y - pad,
                                                 z, n)) / (p_w * p_h);
            if (schedule) {
                forward.compute_root();
            }
        }

        void define_gradients(Func dout, bool schedule = true) {
            check_defined(dout);
            assert(f_input_grads.size() == 0);

            // Gradients for AveragePool not implemented at this time.

            if (schedule) {
            }
        }

        int out_dims() { return 4; }

        int out_dim_size(int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0) {
                // Matching caffe's weird behavior
                size = 1 + std::ceil((float)(in_w + 2 * pad - p_w)/stride);
            } else if (i == 1) {
                // Matching caffe's weird behavior
                size = 1 + std::ceil((float)(in_h + 2 * pad - p_h)/stride);
            } else if (i == 2) {
                size = inputs[0]->out_dim_size(2);
            } else if (i == 3) {
                size = num_samples;
            }
            return size;
        }
};

/*
 * DataLayer is the source data layer for a network (for an image
 * analysis network, the data layer will supply a batch of images)
 *
 * Students will not need to modify the data layer in this assignment
 */
class DataLayer: public Layer {
    public:

        // width and height of images, number of input channels per image,
        // and number of images in the batch
        int in_w, in_h, in_ch, num_samples;

        Var x, y, z, n;
        Image<float> input;

        DataLayer(std::string _name, Image<float> &_input) : Layer(_name, nullptr) {
            input = _input;
            in_w = input.extent(0);
            in_h = input.extent(1);
            in_ch = input.extent(2);
            num_samples = input.extent(3);
            // define forward
            forward(x, y, z, n) = input(x, y, z, n);
        }

        // nothing to propagate
        void define_gradients(Func dout, bool schedule = true) {
            assert(dout.defined());
            return;
        }

        int out_dims() { return 4; }

        int out_dim_size(int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0) {
                size = in_w;
            } else if (i == 1) {
                size = in_h;
            } else if (i == 2) {
                size = in_ch;
            } else if (i == 3) {
                size = num_samples;
            }
            return size;
        }
};

/*
 * Concatenation layer concatencates the values from its two input
 * sources.
 */
class Concat: public Layer {
    public:

        int out_ch;
        int in_w;
        int in_h;
        int num_samples;

        // Halide vars
        Var x, y, z, n;

        Concat(std::string _name, std::vector<Layer*> &in, bool schedule = true) :
               Layer(_name, in) {
            assert(in.size() > 0 && inputs[0]->out_dims() == 4);
            in_w = inputs[0]->out_dim_size(0);
            in_h = inputs[0]->out_dim_size(1);
            num_samples = inputs[0]->out_dim_size(3);

            out_ch = 0;
            for (size_t l = 0; l < in.size(); l++) {
                assert(inputs[l]->out_dim_size(0) == in_w &&
                       inputs[l]->out_dim_size(1) == in_h &&
                       inputs[l]->out_dim_size(3) == num_samples);
                out_ch += inputs[l]->out_dim_size(2);
            }

            // define forward
            forward(x, y, z, n) = cast(inputs[0]->forward.output_types()[0], 0);

            int curr_size = 0;
            for (size_t l = 0; l < inputs.size(); l++) {
                RDom r(0, inputs[l]->out_dim_size(2));
                forward(x, y, curr_size + r.x, n) = inputs[l]->forward(x, y, r.x, n);
                curr_size += inputs[l]->out_dim_size(2);
            }

            if (schedule) {
                forward.compute_root().parallel(n);
            }
        }

        void define_gradients(Func dout, bool schedule = true) {
            check_defined(dout);
            assert(f_input_grads.size() == 0);

            if (schedule) {
            }
        }

        int out_dims() { return 4; }

        int out_dim_size(int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0) {
                size = in_w;
            } else if (i == 1) {
                size = in_h;
            } else if (i == 2) {
                size = out_ch;
            } else if (i == 3) {
                size = num_samples;
            }
            return size;
        }
};

/*
 * Flatten layer takes a multi-dimensional input and "flattens" it
 * into a one dimensional input.  Flattening occurs across space and
 * across channels, but not across samples (images in a batch)
 *
 * Example: an input that was 32x32x4xBATCH_SIZE would be flattened to
 * a 4096xBATCH_SIZE output
 */
class Flatten: public Layer {
    public:
        int out_width;
        int num_samples;

        // Halide vars
        Var x, y, z, n;

        Flatten(std::string _name, Layer *in, bool schedule = true) : Layer(_name, in) {
            assert(in->out_dims() >= 2 && in->out_dims() <= 4);
            num_samples = inputs[0]->out_dim_size(inputs[0]->out_dims() - 1);

            // define forward
            if (inputs[0]->out_dims() == 2) {
                out_width = inputs[0]->out_dim_size(0);
                forward(x, n) = inputs[0]->forward(x, n);
            } else if (inputs[0]->out_dims() == 3) {
                int w = inputs[0]->out_dim_size(0);
                int h = inputs[0]->out_dim_size(1);
                out_width = w * h;
                forward(x, n) = inputs[0]->forward(x%w, x/w, n);
            } else if (inputs[0]->out_dims() == 4) {
                int w = inputs[0]->out_dim_size(0);
                int h = inputs[0]->out_dim_size(1);
                int c = inputs[0]->out_dim_size(2);
                out_width = w * h * c;
                forward(x, n) = inputs[0]->forward(x%w, (x%(w*h))/w, x/(w*h), n);
            }

            if (schedule) {
                forward.compute_root().parallel(n);
            }
        }

        void define_gradients(Func dout, bool schedule = true) {
            check_defined(dout);
            assert(f_input_grads.size() == 0);

            Func in_grad(name + "_in_grad");
            if(inputs[0]->out_dims() == 2)
                in_grad(x, n) = dout(x, n);
            else if(inputs[0]->out_dims() == 3) {
                int w = inputs[0]->out_dim_size(0);
                in_grad(x, y, n) = dout(y*w + x, n);
            } else if (inputs[0]->out_dims() == 4) {
                int w = inputs[0]->out_dim_size(0);
                int h = inputs[0]->out_dim_size(1);
                in_grad(x, y, z, n) = dout(z*w*h + y*w + x, n);
            }
            f_input_grads.push_back(in_grad);

            if (schedule) {
                in_grad.compute_root();
            }
        }

        int out_dims() { return 2; }

        int out_dim_size(int i) {
            assert(i < 2);
            int size = 0;
            if (i == 0) {
                size = out_width;
            } else if (i == 1) {
                size = num_samples;
            }
            return size;
        }
};


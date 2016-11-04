#include <chrono>
#include <random>
#include <unistd.h>
#include "NetworkDefinitions.h"
#include "DataLoaders.h"

/*
 * init_constant --
 *
 * Initializes parameter buffer with the value val.
 */
void init_constant(Image<float> &params, float val) {

    switch (params.dimensions()) {
        case 1:
            for (int i = 0; i < params.extent(0); i++) {
                params(i) = val;
            }
            break;
        case 2:
            for (int i = 0; i < params.extent(0); i++) {
                for (int j = 0; j < params.extent(1); j++) {
                    params(i, j) = val;
                }
            }
            break;
        case 3:
            for (int i = 0; i < params.extent(0); i++) {
                for (int j = 0; j < params.extent(1); j++) {
                    for (int k = 0; k < params.extent(2); k++) {
                        params(i, j, k) = val;
                    }
                }
            }
            break;
        case 4:
            for (int i = 0; i < params.extent(0); i++) {
                for (int j = 0; j < params.extent(1); j++) {
                    for (int k = 0; k < params.extent(2); k++) {
                        for (int l = 0; l < params.extent(3); l++) {
                            params(i, j, k, l) = val;
                        }
                    }
                }
            }
            break;
    }
}

/*
 * init_gaussian --
 *
 * Initializes parameter buffer with random values drawn from Gaussian
 * distribution with given mean and stddev
 */
void init_gaussian(Image<float> &params, float mean, float std_dev,
                   std::random_device &rd) {

    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, std_dev);

    switch (params.dimensions()) {
        case 1:
            for (int i = 0; i < params.extent(0); i++) {
                params(i) = d(gen);
            }
            break;
        case 2:
            for (int i = 0; i < params.extent(0); i++) {
                for (int j = 0; j < params.extent(1); j++) {
                    params(i, j) = d(gen);
                }
            }
            break;
        case 3:
            for (int i = 0; i < params.extent(0); i++) {
                for (int j = 0; j < params.extent(1); j++) {
                    for (int k = 0; k < params.extent(2); k++) {
                        params(i, j, k) = d(gen);
                    }
                }
            }
            break;
        case 4:
            for (int i = 0; i < params.extent(0); i++) {
                for (int j = 0; j < params.extent(1); j++) {
                    for (int k = 0; k < params.extent(2); k++) {
                        for (int l = 0; l < params.extent(3); l++) {
                            params(i, j, k, l) = d(gen);
                        }
                    }
                }
            }
            break;
    }
}

void usage(const char* binary_name) {
    std::cout << "Usage: " << binary_name << " [options] datadir" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -b INT    training mini-batch size" << std::endl;
    std::cout << "  -e INT    num_epochs of training" << std::endl;
    std::cout << "  -m FLOAT  starting momentum" << std::endl;
    std::cout << "  -l FLOAT  starting learning rate" << std::endl;
    std::cout << "  -p FILE   load params from file FILE" << std::endl;
    std::cout << "  -c FILE   save params to file FILE each checkpoint" << std::endl;
    std::cout << "  -f INT    checkpoint every INT mini-batches" << std::endl;
    std::cout << "  -h        this commandline help message" << std::endl;
}

int main(int argc, char **argv) {

    std::string data_path;
    std::string param_file;
    std::string checkpoint_file;

    int   checkpoint_frequency = 1;

    // Training parameters
    int   num_epochs = 1000;
    float learning_rate = 1e-3f;
    float momentum = 0.9f;
    int   batch_size = 32;  // number of samples per mini-batch

    int opt;
    while ((opt = getopt(argc,argv,"b:e:m:l:hp:c:f:")) != EOF) {
        switch(opt) {

            case 'b':
                batch_size = atoi(optarg);
                break;
            case 'e':
                num_epochs = atoi(optarg);
                break;
            case 'm':
                momentum = atof(optarg);
                break;
            case 'l':
                learning_rate = atof(optarg);
                break;
            case 'p':
                param_file = std::string(optarg);
                break;
            case 'c':
               checkpoint_file = std::string(optarg);
               break;
            case 'f':
                checkpoint_frequency = atoi(optarg);
                break;
            case 'h':
            case '?':
            default:
                usage(argv[0]);
                exit(1);
        }
    }

    if (argc <= optind) {
        usage(argv[0]);
        exit(1);
    }

    data_path = argv[optind];

    // true if the code should checkpoint weights to file on disk
    // periodically during training
    bool doCheckpoint = checkpoint_file != std::string("");

    // needed to pull training data from disk
    std::string cifar_training_data_path = data_path + "/cifar-100-binary/train.bin";
    int num_training_images = get_cifar_num_images(cifar_training_data_path);

    /////////////////////////////////////////////////////////////////
    // Set up ToyNet for training
    /////////////////////////////////////////////////////////////////

    ToyNet toy;
    int data_height = 32; // data width
    int data_width = 32; // data height

    toy.define_forward(batch_size, data_width, data_height);

    // this buffer serves to provide the ground truth labels used to
    // compute loss at the start of back-prop.  It will be populated
    // during training with the labels associated with the current
    // mini-batch of images
    Image<int> training_labels(batch_size);
    toy.define_backward(training_labels);

    // buffers to store network loss (a single number) and the
    // per-class score outputs of the softmax layer
    int num_classes = toy.layers["prob"]->out_dim_size(0);
    Image<float> scores(num_classes, batch_size);
    Image<float> loss(1);

    // Initialize the weights of ToyNet.  Initialization is either to
    // random values, or from a parameter definition file that was
    // previously written to disk as a result of prior training.

    if (param_file != std::string("")) {

        // load initial value of model parameters from file

        std::cout << "Loading initial weights from file: " << param_file << std::endl;

        Weights initial_params;
        load_model_from_disk(param_file, initial_params);
        toy.initialize_weights(initial_params);

    } else {

        std::cout << "Using random parameter initialization..." << std::endl;

        // random initialization of network parameters. Weight
        // matrices drawn from a random Gaussian distribution, bias
        // parameter values initialized to 0.
        std::random_device rd;
        init_gaussian(toy.layers["conv1"]->params[0], 0.0f, 0.001, rd);
        init_gaussian(toy.layers["conv1"]->params_cache[0], 0.0f, 0.001, rd);
        init_constant(toy.layers["conv1"]->params[1], 0.0f);
        init_constant(toy.layers["conv1"]->params_cache[1], 0.0f);
        init_gaussian(toy.layers["conv2"]->params[0], 0.0f, 0.001, rd);
        init_gaussian(toy.layers["conv2"]->params_cache[0], 0.0f, 0.001, rd);
        init_constant(toy.layers["conv2"]->params[1], 0.0f);
        init_constant(toy.layers["conv2"]->params_cache[1], 0.0f);
        init_gaussian(toy.layers["fc"]->params[0], 0.0f, 0.001, rd);
        init_gaussian(toy.layers["fc"]->params_cache[0], 0.0f, 0.001, rd);
        init_constant(toy.layers["fc"]->params[1], 0.0f);
        init_constant(toy.layers["fc"]->params_cache[1], 0.0f);
    }

    // create the Halide pipeline that will execute the
    // forward/backward pass to compute gradients.  This pipeline has
    // a large number of outputs since it has to emit per-class
    // scores, overall loss, and gradients for all learnable
    // parameters.

    std::vector<Func> train_outs;

    // overall loss
    Func f_loss = dynamic_cast<SoftMax*>(toy.layers["prob"])->loss(Func(training_labels));
    train_outs.push_back(f_loss);

    // per-class scores
    train_outs.push_back(toy.layers["prob"]->forward);

    // gradients of the various trainable layers
    train_outs.push_back(toy.layers["conv1"]->f_param_grads[0]);
    train_outs.push_back(toy.layers["conv1"]->f_param_grads[1]);
    train_outs.push_back(toy.layers["conv2"]->f_param_grads[0]);
    train_outs.push_back(toy.layers["conv2"]->f_param_grads[1]);
    train_outs.push_back(toy.layers["fc"]->f_param_grads[0]);
    train_outs.push_back(toy.layers["fc"]->f_param_grads[1]);

    // Halide pipeline
    Pipeline training_pipeline(train_outs);

    // Compute the mean of the cifar training data set
    Image<float> mean(data_width, data_height, 3);
    compute_cifar_mean(cifar_training_data_path, mean);

    // get pointer to the network's data layer since its contents will
    // need to be filled in by a mini-batch of images in either
    // iteration of training
    DataLayer *data_layer = dynamic_cast<DataLayer*>(toy.layers["input"]);

    //
    // Everything is set up. Now begin the main training loop here
    //

    // it's fine to not worry about the case where the mini-batch size
    // does not equally divide num_images
    int mini_batches_per_epoch = num_training_images / batch_size;

    std::cout << "Beginning to train network..." << std::endl
              << "   training set size = " << num_training_images << std::endl
              << "   mini-batch size   = " << batch_size  << std::endl
              << "   num_epochs        = " << num_epochs << std::endl
              << "   learning_rate     = " << learning_rate  << std::endl
              << "   momentum          = " << momentum << std::endl << std::endl;

    // main training loop
    for (int epoch = 0; epoch<num_epochs; epoch++) {

        for (int mb = 0; mb < mini_batches_per_epoch; mb++) {

            auto start = std::chrono::steady_clock::now();

            // load a batch of images from the cifar database for
            // training.  Loads a random sample of batch_size images
            // and populates the network's input buffer (data_layer->input)
            // and the training labels buffer (training_labels).  Once this
            // data is loaded into these buffers, the network is read to
            // perform and forward/backward pass to compute gradients from
            // this minibatch
            load_cifar_batch_random(cifar_training_data_path, batch_size, mean,
                                    data_layer->input, training_labels);

            // run forward evaluation and back-prop to compute gradients
            training_pipeline.realize(
                {loss, scores,
                 toy.layers["conv1"]->param_grads[0],
                 toy.layers["conv1"]->param_grads[1],
                 toy.layers["conv2"]->param_grads[0],
                 toy.layers["conv2"]->param_grads[1],
                 toy.layers["fc"]->param_grads[0],
                 toy.layers["fc"]->param_grads[1]
                });

            auto end = std::chrono::steady_clock::now();

            // Now update the network weights given the gradiants
            // computed from the minibatch
            for (auto l: toy.layers) {
                int num_params = l.second->params.size();
                for (int p = 0; p < num_params; p++) {

                    // TODO: students should update parameters using momentum here
                    //
                    // Given parameter buffer: l.second->params[p]
                    //   -- Gradient buffer: l.second->params_grads[p]
                    //   -- Old parameter:   l.second->update_cache[p]
                    //
                    // And values for momentum and learning rate
                    // Perform a parameter update step of gradiant descent with momentum
                }
            }

            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // print status
            std::cout << "Epoch=" << epoch << ", mb=" << mb << "/" << mini_batches_per_epoch
                      << " loss: " << loss(0) << std::endl;
            std::cout << "Mini-batch processing time: " << elapsed << "ms "
                      << (float)elapsed / batch_size << " ms/image)" << std::endl;

            // potentially checkpoint the network's weights
            if (doCheckpoint && (mb % checkpoint_frequency == 0) ) {

                std::ostringstream oss;
                oss << checkpoint_file << ".ep-" << epoch << ".mb-" << mb;
                std::string filename = oss.str();

                Weights weights;
                toy.extract_weights(weights);
                save_model_to_disk(filename, weights);

                std::cout << "Wrote checkpoint: " << filename << std::endl;
            }
        }
    }

    std::cout << "Completed training!" << std::endl;

    //
    // Now verify the accuracy of your network.
    //

    std::cout << "Running inference on test set to check accuracy of the trainined network..." << std::endl;

    std::string cifar_test_data_path = data_path + "/cifar-100-binary/test.bin";
    int num_test_images = get_cifar_num_images(cifar_test_data_path);

    std::vector<Func> test_outs;
    test_outs.push_back(toy.layers["prob"]->forward);
    Pipeline test_pipeline(test_outs);

    size_t num_batches = num_test_images/batch_size;
    int num_total = 0;
    int num_correct = 0;
    for (size_t b_id = 0; b_id < num_batches; b_id++) {

        auto start = std::chrono::steady_clock::now();

        int image_index = b_id * batch_size;
        load_cifar_batch(cifar_test_data_path, batch_size, image_index, mean,
                         data_layer->input, training_labels);

        test_pipeline.realize({scores});

        int max_index = std::min(batch_size, num_test_images - image_index);
        int n = 0;

        // for all images in the batch, check the predicted class
        // label against the ground truth class label
        for (int l = 0; l < max_index; l++) {

            // find max score
            int best_class = 0;
            float best_score = 0.0f;
            for (int c = 0; c < scores.extent(0); c++) {
                if (scores(c, n) > best_score) {
                    best_score = scores(c, n);
                    best_class = c;
                }
            }

            // check against ground truth
            if (best_class == training_labels(l)) {
                num_correct++;
            }

            num_total++;
            n++;
        }

        auto end = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Batch time (inference): " << elapsed
            << "ms" << " (" <<  ((float)elapsed / batch_size) << " ms/image)" << std::endl;
    }

    std::cout << "Test set accuracy: " << (float)num_correct/num_total << std::endl;

    return 0;
}

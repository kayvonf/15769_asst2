#include <chrono>
#include <unistd.h>
#include "NetworkDefinitions.h"
#include "DataLoaders.h"


void usage(const char* binary_name) {
    std::cout << "Usage: " << binary_name << " [options] modeldir datadir network" << std::endl;
    std::cout << std::endl;
    std::cout << "   Valid network names: vgg, inception" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -b INT    batch size" << std::endl;
    std::cout << "  -n INT    number of images to test on" << std::endl;
    std::cout << "  -v        verbose mode" << std::endl;
    std::cout << "  -h        this commandline help message" << std::endl;
}

int main(int argc, char **argv) {

    std::string model_path, data_path, net_name;
    int batch_size = 1;
    int max_test_images = -1;
    bool verbose = false;

    int opt;
    while ((opt = getopt(argc,argv,"b:n:vh")) != EOF) {
        switch(opt) {
            case 'n':
                max_test_images = atoi(optarg);
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
            case '?':
            default:
                usage(argv[0]);
                exit(1);
        }
    }

    if (argc <= optind+2) {
        usage(argv[0]);
        exit(1);
    }

    model_path = argv[optind];
    data_path = argv[optind+1];
    net_name = argv[optind+2];

    if (!(net_name == "vgg" || net_name == "inception")) {
        std::cout << "Network name must be either vgg or inception" << std::endl;
        exit(1);
    }

    Weights weights;

    int data_height = 224;
    int data_width = 224;

    Network *net = nullptr;
    std::string weight_path;
    std::string reference_path;
    if (net_name == "vgg") {
        weight_path = model_path + "/vgg.bin";
        reference_path = data_path + "/vgg_reference_results.txt";
        net = new Vgg();
    } else {
        weight_path = model_path + "/googlenet.bin";
        reference_path = data_path + "/googlenet_reference_results.txt";
        net = new GoogleNet();
    }

    std::cout << "Loading weights from disk..." << std::endl;
    load_model_from_disk(weight_path, weights);

    std::cout << "Building DNN pipeline..." << std::endl;

    net->define_forward(batch_size, data_width, data_height);
    net->initialize_weights(weights);

    // Note to students about the implementation below: During testing
    // we found that larger networks with 100's stages get too big for
    // Halide to compile successfully (deficiency in the current
    // compiler implementation).  As a result, for larger networks
    // like GoogleNet in the network's definition, we manually break
    // the pipeline into multiple, smaller "sub-pipelines" and feed
    // the output of the first into the latter.  Essentially this is
    // equivalent to forcing a compute_root() between the
    // subpipelines.  This implementation detail is relevant to
    // students wishing to perform optimizations that may cross
    // pipeline stages (cross-layer fusion) -- since you can't fuse
    // across two distinct Halide pipelines.
    //
    // Vgg and ToyNet all consist of a single subpipeline so this
    // detail doesn't matter. GoogleNet is split into a few
    // subpipelines.
    std::vector<Pipeline> sub_pipelines;
    std::vector<Image<float>> sub_pipeline_outs;
    DataLayer *dl = nullptr;
    for (size_t p = 0; p < net->sub_pipeline_end_points.size(); p++) {
        std::pair<std::string, std::string> end_point = net->sub_pipeline_end_points[p];
        Layer *out_layer = net->layers[end_point.first];
        if (end_point.second == "input") {
            assert(p == 0);
            dl = dynamic_cast<DataLayer*>(net->layers["input"]);
        }

        if (p > 0) {
            DataLayer *inter = dynamic_cast<DataLayer*>(net->layers[end_point.second]);
            assert(inter != nullptr);
            sub_pipeline_outs.push_back(inter->input);
        }

        std::vector<Func> outs;
        outs.push_back(out_layer->forward);
        Pipeline test(outs);
        sub_pipelines.push_back(test);
    }

    assert(dl);

    Layer *softm = net->layers["prob"];
    int num_classes = softm->out_dim_size(0);
    Image<float> scores(num_classes, batch_size);
    sub_pipeline_outs.push_back(scores);

    assert(sub_pipeline_outs.size() == sub_pipelines.size());

    // compile the Halide pipeline (technically compile all the subpipelines)
    for (size_t p = 0; p < sub_pipelines.size(); p++) {
        auto compile_start = std::chrono::steady_clock::now();

        sub_pipelines[p].compile_jit();

        auto compile_end = std::chrono::steady_clock::now();
        std::cout << "Sub-pipeline " << p << " compile time: " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(compile_end - compile_start).count()
            << "ms" << std::endl;
    }

    //
    // Now that the network has been compiled, evaluate the network on
    // all images in the test set, measuring both the accuracy and the
    // performance of the network
    //

    std::string label_file_name = data_path + "/imagenet_labels.txt";
    std::vector<std::string> test_image_names;
    std::vector<int> image_labels;
    get_image_names_and_labels(label_file_name, test_image_names, image_labels);

    std::vector<int> reference_labels;
    std::vector<float> reference_scores;
    get_reference_labels(reference_path, reference_labels, reference_scores);

    std::string image_dir = data_path + "/mini_val";

    int num_images = (max_test_images == -1) ?  test_image_names.size() : max_test_images;
    size_t num_batches = num_images / batch_size;
    int num_total = 0;
    int num_correct = 0;
    int num_reference_matches = 0;
    float total_elapsed = 0.f;

    std::cout << "Testing using " << num_batches * batch_size << " images (batch size: "
              << batch_size << ")" << std::endl;

    for (size_t b_id = 0; b_id < num_batches; b_id++) {

        auto start = std::chrono::steady_clock::now();

        size_t image_index = b_id * batch_size;
        load_imagenet_batch(test_image_names, image_dir, image_index, true, dl->input);

        for (size_t p = 0; p < sub_pipelines.size(); p++) {
            sub_pipelines[p].realize(sub_pipeline_outs[p]);
        }

        int max_index = std::min(image_index + batch_size, image_labels.size());
        int n = 0;

        // for all images in the batch, check the predicted class
        // label against the ground truth class label
        for (int l = image_index; l < max_index; l++) {

            // find max score
            int best_class = 0;
            float best_score = 0.0f;
            for (int c = 0; c < scores.extent(0); c++) {
                if (scores(c, n) > best_score) {
                    best_score = scores(c, n);
                    best_class = c;
                }
            }

            if (verbose) {
                std::cout << "Image " << l << ":"
                          << " predicted: " << best_class
                          << " score: " << best_score << std::endl;
            }

            // check against ground truth
            if (best_class == image_labels[l]) {
                num_correct++;
            }

            // check against output of class reference implementation
            if (best_class == reference_labels[l]) {
                num_reference_matches++;
            } else if (verbose) {
                std::cout << "WARNING: DNN output DOES NOT match staff reference (" << l << ")" << std::endl;
            }

            num_total++;
            n++;
        }

        auto end = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        total_elapsed += elapsed;
        if (verbose)
            std::cout << "Batch time: " << elapsed
                      << "ms" << " (" <<  ((float)elapsed / batch_size) << " ms/image)" << std::endl;
    }

    if (net) {
        delete net;
    }

    //
    // report final stats
    //

    std::cout << total_elapsed / (num_batches * batch_size) << " ms/image" << std::endl;
    std::cout << "DNN Accuracy: " << 100.f * (float)num_correct/num_total << "\%" << std::endl;
    if (num_reference_matches == (int)(num_batches * batch_size)) {
        std::cout << "Network output matches the reference (CORRECT)" << std::endl;
    } else {
        std::cout << "Network output matches reference on "
                  << num_reference_matches << " of " << (num_batches * batch_size) << std::endl;
    }

    return 0;
}

#include <csv.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <algorithm>
#include <random>
#include "Perceptron.hpp"
#include "AdaptiveLinearNeuron.hpp"

using namespace csv;

// convert if we can, else return flt_min
float ConvStrToFlt(const std::string& s) {
    std::istringstream iss(s);
    float f;
    iss >> std::noskipws >> f;
    return (iss.eof() && !iss.fail() ? f : -FLT_MAX); // 
}

float program() {
    MultiClassifier<Adaline> classifier;
    std::vector<Sample> data_set;

    CSVReader reader("iris.csv");
    for (CSVRow& row : reader) {
        Sample sample;

        for (int i = 0; i < row.size(); i++) {
            std::string  p = row[i].get<>();

            if (i + 1 != row.size()) {
                sample.factors.push_back(ConvStrToFlt(p));
            } else {
                int label = classifier.GetID(p);
                sample.label = label;
            }
        }

        data_set.push_back(sample);
    }

    // normalize data set
    int num_factors = data_set.back().factors.size();
    std::vector<float> mean(num_factors);
    std::fill(mean.begin(), mean.end(), 0.0f);

    for (const Sample& s : data_set) {
        for (int i = 0; i < num_factors; i++) {
            mean[i] += s.factors[i];
        }
    }

    for (int i = 0; i < num_factors; i++) {
        mean[i] /= data_set.size();
    }

    std::vector<float> stddev(num_factors);
    std::fill(stddev.begin(), stddev.end(), 0.0f);

    for (const Sample& s : data_set) {
        for (int i = 0; i < num_factors; i++) {
            float diff = s.factors[i] - mean[i];
            stddev[i] += diff * diff;
        }
    }
    
    for (int i = 0; i < num_factors; i++) {
        stddev[i] = sqrt(stddev[i] / (data_set.size() - 1));
    }

    for (Sample& s : data_set) {
        for (int i = 0; i < num_factors; i++) {
            s.factors[i] = (s.factors[i] - mean[i]) / stddev[i];
        }
    }

    // now that we have our data, let's use 50% of it train
    std::shuffle(data_set.begin(), data_set.end(), std::default_random_engine(rand()));

    auto select_iter = data_set.begin() + (data_set.size() / 3);

    std::vector<Sample> training_set, test_set;
    std::copy(data_set.begin(), select_iter, std::back_inserter(training_set));
    std::copy(select_iter, data_set.end(), std::back_inserter(test_set));

    classifier.Train(training_set);

    // test how well our model performs
    int misclassifications = 0;
    for (const Sample& sample : test_set) {
        if (classifier.PredictClass(sample) != sample.label) {
            misclassifications++;
        }
    }

    float mcr = (float)misclassifications / test_set.size();
    std::cout << "Misclassification rate: " << mcr << '\n';
    return mcr;
}

int main() {
    srand(time(nullptr));

    int num_tests = 128;
    float mcr = 0.0f;
    for (int i = 0; i < num_tests; i++) {
        mcr += program();
    }
    std::cout << "Overall error " << mcr / num_tests << '\n';
}
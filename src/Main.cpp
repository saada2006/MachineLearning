#include <csv.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <algorithm>
#include <random>

constexpr float learning_rate = 0.01f;

using namespace csv;

// convert if we can, else return flt_min
float ConvStrToFlt(const std::string& s) {
    std::istringstream iss(s);
    float f;
    iss >> std::noskipws >> f;
    return (iss.eof() && !iss.fail() ? f : FLT_MIN); // 
}

struct Sample {
    std::vector<float> factors;
    int label;

    float get_input(int x) const {
        if (x == 0) {
            return 1.0f;
        }
        else {
            return factors[x - 1];
        }
    }
};

int classify(const std::vector<float>& weights, const Sample& sample) {
    float z = 0.0f;
    for (int k = 0; k < weights.size(); k++) {
        z += weights[k] * sample.get_input(k);
    }
    int classification = (z < 0.0f ? -1 : 1);
    return classification;
}

int main() {
    CSVReader reader("iris.csv");

    std::vector<Sample> data_set;

    std::map<std::string, int> class_ids;
    int next_id = 1;


    for (CSVRow& row : reader) {
        Sample sample;
        bool binary = true;
        for (CSVField& field : row) {
            std::string  prop = field.get<>();
            float f = ConvStrToFlt(prop);
            if (f != FLT_MIN) {
                sample.factors.push_back(f);
            }
            else {
                int& label = class_ids[prop];
                if (label == 0) {
                    label = next_id++;
                }
                if (label == 3) {
                    binary = false;
                    break;
                }
                sample.label = (label == 1 ? -1 : 1);
            }
        }

        if (!binary) {
            continue;
        }

        data_set.push_back(sample);
    }

    // now that we have our data, let's use 50% of it train
    std::shuffle(data_set.begin(), data_set.end(), std::default_random_engine(std::time(nullptr)));

    auto select_iter = data_set.begin() + (data_set.size() / 2);

    std::vector<Sample> training_set, test_set;

    std::copy(data_set.begin(), select_iter, std::back_inserter(training_set));
    std::copy(select_iter, data_set.end(), std::back_inserter(test_set));

    // train a perceptron

    // create our weights
    std::vector<float> weights(data_set.front().factors.size() + 1);
    for (float f : weights) {
        f = 0.0f; // init to zero
    }
    

    // adjust the weights
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < training_set.size(); j++) {
            // compute the result and misclassification error
            int classification = classify(weights, training_set[j]);

            // compute adjustment
            for (int k = 0; k < weights.size(); k++) {
                weights[k] += learning_rate * (training_set[j].label - classification) * training_set[j].get_input(k);
            }
        }
    }

    // test how well our model performs


    int misclassifications = 0;

    for (const Sample& sample : test_set) {
        if (classify(weights, sample) != sample.label) {
            misclassifications++;
        }
    }

    std::cout << "Misclassification rate: " << (float)misclassifications / test_set.size() << '\n';


}
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
    return (iss.eof() && !iss.fail() ? f : -FLT_MAX); // 
}

struct Sample {
    std::vector<float> factors;
    int label;

    float GetInput(int x) const {
        if (x == 0) {
            return 1.0f;
        }
        else {
            return factors[x - 1];
        }
    }
};

class BinaryClassifier {
public:
    virtual float PredictChance(const Sample& sample) = 0; // phi(z) func
    virtual bool PredictBinary(const Sample& sample) = 0; // binary value
    virtual int PredictClass(const Sample& sample) = 0; // pos or neg class

    virtual void Train(const std::vector<Sample>& training_set) = 0;
protected:
    std::vector<float> weights;
};

class Perceptron : public BinaryClassifier {
public:
    virtual float PredictChance(const Sample& sample) {
        float z = 0.0f;
        for (int k = 0; k < weights.size(); k++) {
            z += weights[k] * sample.GetInput(k);
        }
        return z;
    }

    virtual bool PredictBinary(const Sample& sample) {
        return (PredictChance(sample) < 0.0f ? false : true);
    }

    virtual int PredictClass(const Sample& sample) {
        return  (PredictChance(sample) < 0.0f ? -1 : 1);
    }

    virtual void Train(const std::vector<Sample>& training_set) {
        weights.resize(training_set.back().factors.size() + 1);
        std::fill(weights.begin(), weights.end(), 0.0f);

        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < training_set.size(); j++) {
                // compute the result and misclassification error
                int classification = PredictClass(training_set[j]);

                // compute adjustment
                for (int k = 0; k < weights.size(); k++) {
                    weights[k] += learning_rate * (training_set[j].label - classification) * training_set[j].GetInput(k);
                }
            }
        }
    }
};

template<class T>
class MultiClassifier {
public:
    MultiClassifier() : num_ids(0) {}

    int GetID(const std::string& s) {
        int id;
        auto iter = class_ids.find(s);
        if (iter == class_ids.end()) {
            id = num_ids++;
            class_ids.emplace(s, id);
        }
        else {
            id = iter->second;
        }
        return id;
    }

    void Train(const std::vector<Sample>& training_set) {
        int num_factors = training_set.back().factors.size();
        // for each id, create a binary classifier that determines the sample as "part of class x" or "not part of class x"
        for (int i = 0; i < num_factors; i++) {
            std::vector<Sample> onevall_set;
            for (const Sample& s : training_set) {
                Sample binclass = s;
                binclass.label = (binclass.label == i ? 1 : -1);
                onevall_set.push_back(binclass);
            }

            T binclassifier;
            binclassifier.Train(onevall_set);

            classifiers.push_back(binclassifier);
        }
    }

    int PredictClass(const Sample& s) {
        float best_chance = -FLT_MAX;
        int best_class = 0;
        for (int i = 0; i < num_ids; i++) {
            float z = classifiers[i].PredictChance(s);
            if (z > best_chance) {
                best_chance = z;
                best_class = i;
            }
        }
        return best_class;
    }
private:
    int num_ids;
    std::map<std::string, int> class_ids;
    std::vector<T> classifiers;
};

int main() {
    CSVReader reader("iris.csv");

    std::vector<Sample> data_set;

    MultiClassifier<Perceptron> classifier;

    for (CSVRow& row : reader) {
        Sample sample;
        for (CSVField& field : row) {
            std::string  prop = field.get<>();

            float f = ConvStrToFlt(prop);
            if (f != -FLT_MAX) {
                sample.factors.push_back(f);
            }
            else {
                int label = classifier.GetID(prop);
                sample.label = label;
            }
        }

        data_set.push_back(sample);
    }

    // now that we have our data, let's use 50% of it train
    std::shuffle(data_set.begin(), data_set.end(), std::default_random_engine(std::time(nullptr)));

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

    std::cout << "Misclassification rate: " << (float)misclassifications / test_set.size() << '\n';


}
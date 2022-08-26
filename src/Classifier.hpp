#pragma once

#include <vector>
#include <map>
#include "Sample.hpp"
#include "Parameters.hpp"

class BinaryClassifier {
public:
    virtual float PredictChance(const Sample& sample) = 0; // phi(z) func
    virtual bool PredictBinary(const Sample& sample) = 0; // binary value
    virtual int PredictClass(const Sample& sample) = 0; // pos or neg class

    virtual void Train(const std::vector<Sample>& training_set) = 0;
protected:
    std::vector<float> weights;
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
        std::vector<Sample> onevall_set = training_set;
        for (int i = 0; i < num_factors; i++) {
            for (int j = 0; j < training_set.size(); j++) {
                onevall_set[j].label = (training_set[j].label == i ? 1 : -1);
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
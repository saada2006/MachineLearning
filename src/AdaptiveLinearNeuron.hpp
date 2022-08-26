#pragma once
#include "Classifier.hpp"

// one interesting observation we can make is that this is literally the perceptron except using phi(z) = z instead of a weird step function
class Adaline : public BinaryClassifier {
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


        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < training_set.size(); j++) {
                // compute the result and misclassification error
                float z = PredictChance(training_set[j]);

                // compute adjustment
                for (int k = 0; k < weights.size(); k++) {
                    weights[k] += CalcLearningRate(i) * (training_set[j].label - z) * training_set[j].GetInput(k);
                }
            }
        }
    }


};
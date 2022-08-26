#pragma once

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
#pragma once

constexpr float c1 = 1.0f;
constexpr float c2 = 10.0f;

constexpr int epochs = 1;

float CalcLearningRate(int n) {
    return 0.01f;
    return c1 / (n + c2);
}
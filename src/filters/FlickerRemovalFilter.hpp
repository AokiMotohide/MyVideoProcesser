#pragma once
#include "VideoFilter.hpp"
#include <deque>
#include <vector>

// Implementation of Luminance Smoothing (Moving Average)
class FlickerRemovalFilter : public VideoFilter {
private:
    int windowSize;
    float strength;

    // A structure to hold the average luminance of past and future frames
    std::deque<double> luminanceBuffer;

    // Helper functions
    double calculateFrameLuminance(const cv::Mat& frame);
    cv::Mat adjustLuminance(const cv::Mat& frame, double targetLuminance, double currentLuminance);

    // To handle true moving average, ideally we need lookahead,
    // but for real-time/streaming constraints, we often use a delayed buffer.
    // In this basic version, we will compute the average of the *last N frames* + *current frame*
    // For a better offline result, we could look ahead, but that complicates the simple pipeline.
    // We will stick to a causal filter (using only past and current frames) for simplicity,
    // blending the current luminance towards the moving average.

public:
    FlickerRemovalFilter();

    cv::Mat apply(const cv::Mat& frame, int frameIndex, VideoContext& ctx) override;
    
    void init(const VideoContext& ctx) override;

    void setStrength(float value) override;
    
    // Set how many past frames to average over
    void setWindowSize(int size);
    int getWindowSize() const;

    std::string getName() const override;
};

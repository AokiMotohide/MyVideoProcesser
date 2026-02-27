#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// Context passed to filters, useful for cross-frame information if needed
struct VideoContext {
    int totalFrames;
    int currentFrame;
    double fps;
    cv::Size frameSize;
};

// Base interface for all video filters
class VideoFilter {
public:
    virtual ~VideoFilter() = default;

    // Apply the filter to the given frame. Should return the processed frame.
    virtual cv::Mat apply(const cv::Mat& frame, int frameIndex, VideoContext& ctx) = 0;

    // Optional initialization before processing starts
    virtual void init(const VideoContext& ctx) {}

    // Set the strength of the filter (0.0 to 1.0)
    virtual void setStrength(float value) = 0;

    // Get the name of the filter
    virtual std::string getName() const = 0;
};

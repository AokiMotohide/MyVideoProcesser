#include "VideoProcessor.hpp"
#include <iostream>

VideoProcessor::VideoProcessor() : isProcessing(false), stopRequested(false) {
  context.totalFrames = 0;
  context.currentFrame = 0;
  context.fps = 30.0;
}

VideoProcessor::~VideoProcessor() {
  if (capture.isOpened())
    capture.release();
  if (writer.isOpened())
    writer.release();
}

bool VideoProcessor::loadInput(const std::string &path) {
  inputPath = path;
  if (capture.isOpened())
    capture.release();

  capture.open(inputPath);
  if (!capture.isOpened()) {
    std::cerr << "Failed to open video source: " << path << std::endl;
    return false;
  }

  context.totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
  context.fps = capture.get(cv::CAP_PROP_FPS);
  context.frameSize =
      cv::Size(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
               static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));

  // For image sequences, total info might not be immediately available
  if (context.totalFrames <= 0)
    context.totalFrames = 10000; // Arbitrary large number or user defined

  context.currentFrame = 0;
  return true;
}

bool VideoProcessor::setOutput(const std::string &path) {
  outputPath = path;
  return true;
}

void VideoProcessor::addFilter(std::shared_ptr<VideoFilter> filter) {
  if (filter) {
    filters.push_back(filter);
  }
}

void VideoProcessor::clearFilters() { filters.clear(); }

bool VideoProcessor::process() {
  if (!capture.isOpened() || outputPath.empty())
    return false;

  // Re-open capture to start from beginning
  capture.set(cv::CAP_PROP_POS_FRAMES, 0);
  context.currentFrame = 0;

  int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  // For image sequences outputs, writer setup might differ or not use
  // VideoWriter. Assuming video output for now.
  writer.open(outputPath, codec, context.fps, context.frameSize, true);

  if (!writer.isOpened()) {
    std::cerr << "Failed to open output video: " << outputPath << std::endl;
    return false;
  }

  // Initialize filters
  for (auto &f : filters) {
    f->init(context);
  }

  isProcessing = true;
  stopRequested = false;

  cv::Mat frame;
  while (!stopRequested && capture.read(frame)) {
    if (frame.empty())
      break;

    cv::Mat processed = frame;

    // Apply chain of filters
    for (auto &f : filters) {
      processed = f->apply(processed, context.currentFrame, context);
    }

    writer.write(processed);
    context.currentFrame++;
  }

  writer.release();
  isProcessing = false;

  return !stopRequested;
}

cv::Mat VideoProcessor::processPreviewFrame(int frameIndex) {
  if (!capture.isOpened())
    return cv::Mat();

  capture.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
  cv::Mat frame;
  capture.read(frame);

  if (frame.empty())
    return frame;

  // Create a temporary context for preview
  VideoContext tempCtx = context;
  tempCtx.currentFrame = frameIndex;

  cv::Mat processed = frame;
  for (auto &f : filters) {
    processed = f->apply(processed, frameIndex, tempCtx);
  }

  return processed;
}

void VideoProcessor::stop() { stopRequested = true; }

float VideoProcessor::getProgress() const {
  if (context.totalFrames > 0) {
    return static_cast<float>(context.currentFrame) / context.totalFrames;
  }
  return 0.0f;
}

int VideoProcessor::getTotalFrames() const { return context.totalFrames; }
int VideoProcessor::getCurrentFrame() const { return context.currentFrame; }
bool VideoProcessor::getIsProcessing() const { return isProcessing; }
double VideoProcessor::getFPS() const { return context.fps; }
cv::Size VideoProcessor::getFrameSize() const { return context.frameSize; }

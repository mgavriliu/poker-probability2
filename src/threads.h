#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>

class ThreadPool {
 public:
  ThreadPool(size_t numThreads);
  ~ThreadPool();
  void enqueue(std::function<void()> task);

 private:
  std::vector<std::thread>          workers;
  std::deque<std::function<void()>> tasks;
  std::mutex                        queueMutex;
  std::condition_variable           condition;
  bool                              stop;
};
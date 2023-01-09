// C++ example
#include <iostream>
#include <chrono>

#define LOOP_COUNT 1000000

int main(int argc, char** argv) {
  using namespace std::chrono;
  long total_time = 0;

  for (int i = 0; i < LOOP_COUNT; i++) {
    high_resolution_clock::time_point start = high_resolution_clock::now();

    // Define task here
    int result = 0;
    for (int j = 0; j < 100; j++) {
      result += j;
    }

    high_resolution_clock::time_point end = high_resolution_clock::now();
    long elapsed_time = duration_cast<nanoseconds>(end - start).count();
    total_time += elapsed_time;
  }

  double average_time = (double)total_time / (double)LOOP_COUNT;
  std::cout << "Average time: " << average_time << " nanoseconds" << std::endl;

  return 0;
}

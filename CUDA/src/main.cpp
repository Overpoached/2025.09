#include <iostream>
#include "kernel.h"

int main() {
    std::cout << "=== CUDA Array Addition Demo ===" << std::endl;

    PrintDeviceProperties();

    // Create an array of integers
    const int SIZE = 1000;
    int* data = new int[SIZE];

    // Initialize array with values 0 to 999
    std::cout << "\nInitializing array with values 0 to " << (SIZE - 1) << std::endl;
    for (int i = 0; i < SIZE; i++) {
        data[i] = i;
    }

    // Calculate expected sum before addition (0 + 1 + 2 + ... + 999)
    int expectedSumBefore = (SIZE - 1) * SIZE / 2;
    std::cout << "Sum of original array: " << expectedSumBefore << std::endl;

    // Print first 10 elements before
    std::cout << "\nFirst 10 elements before GPU processing: ";
    for (int i = 0; i < 10; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // Send data to GPU, add 1 to each element, and get sum
    std::cout << "\nSending data to GPU for processing..." << std::endl;
    int sum = addOneAndSum(data, SIZE);

    // Print first 10 elements after
    std::cout << "\nFirst 10 elements after GPU processing: ";
    for (int i = 0; i < 10; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // Display results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Sum of array after adding 1 to each element: " << sum << std::endl;
    std::cout << "Expected sum: " << (expectedSumBefore + SIZE) << std::endl;

    // Verify correctness
    if (sum == expectedSumBefore + SIZE) {
        std::cout << "\n✓ SUCCESS: GPU computation is correct!" << std::endl;
    } else {
        std::cout << "\n✗ ERROR: GPU computation failed!" << std::endl;
    }

    // Cleanup
    delete[] data;

    return 0;
}

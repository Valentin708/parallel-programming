#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

using Matrix = std::vector<double>;

__global__ void matmulKernel(const double *A, const double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

bool readMatrix(const std::string& filename, Matrix& data, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
        return false;
    }
    file >> n;
    if (file.fail() || n <= 0) {
        std::cerr << "Ошибка: неверный формат файла" << std::endl;
        return false;
    }
    data.resize(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file >> data[i * n + j];
            if (file.fail()) {
                std::cerr << "Ошибка: недостаточно данных" << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool writeMatrix(const std::string& filename, const Matrix& data, int n) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось создать файл " << filename << std::endl;
        return false;
    }
    file << n << "\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << std::fixed << std::setprecision(10) << data[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << "\n";
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::string fileA = "matrix1.txt";
    std::string fileB = "matrix2.txt";
    std::string fileC = "result.txt";
    int block_size = 32;

    if (argc >= 3) {
        fileA = argv[1];
        fileB = argv[2];
        if (argc >= 4) fileC = argv[3];
        if (argc >= 5) block_size = std::stoi(argv[4]);
    }

    std::cout << "Чтение матрицы A из " << fileA << " ..." << std::endl;
    Matrix A;
    int n;
    if (!readMatrix(fileA, A, n)) return 1;

    std::cout << "Чтение матрицы B из " << fileB << " ..." << std::endl;
    Matrix B;
    if (!readMatrix(fileB, B, n)) return 1;

    std::cout << "Размер матриц: " << n << " x " << n << std::endl;
    std::cout << "Размер блока: " << block_size << " x " << block_size << std::endl;

    dim3 threadsPerBlock(block_size, block_size);
    dim3 blocksPerGrid(
        (n + block_size - 1) / block_size,
        (n + block_size - 1) / block_size
    );

    double *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(double);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    Matrix C(n * n, 0.0);
    auto start = std::chrono::high_resolution_clock::now();

    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Время умножения на GPU: " << std::fixed << std::setprecision(6) << elapsed.count() << " с" << std::endl;

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (writeMatrix(fileC, C, n)) {
        std::cout << "Результат записан в " << fileC << std::endl;
    } else {
        return 1;
    }

    return 0;
}
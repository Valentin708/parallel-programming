#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>

using Matrix = std::vector<double>;

bool readMatrix(const std::string& filename, Matrix& data, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
        return false;
    }

    file >> n;
    if (file.fail() || n <= 0) {
        std::cerr << "Ошибка: неверный формат файла (ожидалось положительное целое число)" << std::endl;
        return false;
    }

    data.resize(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file >> data[i * n + j];
            if (file.fail()) {
                std::cerr << "Ошибка: недостаточно данных в файле " << filename << std::endl;
                return false;
            }
        }
    }
    return true;
}

void multiply(const Matrix& A, const Matrix& B, Matrix& C, int n) {
    C.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) { 
            double aik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
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

    if (argc >= 3) {
        fileA = argv[1];
        fileB = argv[2];
        if (argc >= 4) fileC = argv[3];
    }

    std::cout << "Чтение матрицы A из " << fileA << " ..." << std::endl;
    Matrix A;
    int nA;
    if (!readMatrix(fileA, A, nA)) return 1;

    std::cout << "Чтение матрицы B из " << fileB << " ..." << std::endl;
    Matrix B;
    int nB;
    if (!readMatrix(fileB, B, nB)) return 1;

    if (nA != nB) {
        std::cerr << "Ошибка: размеры матриц не совпадают (" << nA << " != " << nB << ")" << std::endl;
        return 1;
    }
    int n = nA;

    std::cout << "Размер матриц: " << n << " x " << n << std::endl;
    std::cout << "Объём задачи (количество элементов): " << (n * n) << " в каждой матрице" << std::endl;

    Matrix C;
    auto start = std::chrono::high_resolution_clock::now();
    multiply(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Время умножения: " << std::fixed << std::setprecision(6) << elapsed.count() << " с" << std::endl;

    if (writeMatrix(fileC, C, n)) {
        std::cout << "Результат записан в " << fileC << std::endl;
    } else {
        return 1;
    }

    return 0;
}
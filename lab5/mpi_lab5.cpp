#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <mpi.h>

using Matrix = std::vector<double>;

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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string fileA = "matrix1.txt";
    std::string fileB = "matrix2.txt";
    std::string fileC = "result.txt";

    if (argc >= 3) {
        fileA = argv[1];
        fileB = argv[2];
        if (argc >= 4) fileC = argv[3];
    }

    int n = 0;
    Matrix A, B;

    if (rank == 0) {
        std::cout << "Чтение матрицы A из " << fileA << " ..." << std::endl;
        if (!readMatrix(fileA, A, n)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        std::cout << "Чтение матрицы B из " << fileB << " ..." << std::endl;
        if (!readMatrix(fileB, B, n)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        std::cout << "Размер матриц: " << n << " x " << n << std::endl;
        std::cout << "Количество процессов: " << size << std::endl;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    Matrix A_local(local_rows * n);
    Matrix C_local(local_rows * n, 0.0);

    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int p_start = p * rows_per_proc + (p < remainder ? p : remainder);
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            MPI_Send(&A[p_start * n], p_rows * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
        std::copy(A.begin() + start_row * n, A.begin() + (start_row + local_rows) * n, A_local.begin());
    } else {
        MPI_Recv(A_local.data(), local_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            MPI_Send(B.data(), n * n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
    } else {
        B.resize(n * n);
        MPI_Recv(B.data(), n * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A_local[i * n + k];
            for (int j = 0; j < n; ++j) {
                C_local[i * n + j] += aik * B[k * n + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Время умножения (максимальное среди процессов): " 
                  << std::fixed << std::setprecision(6) << max_time << " с" << std::endl;
    }

    Matrix C(n * n, 0.0);
    if (rank == 0) {
        std::copy(C_local.begin(), C_local.end(), C.begin() + start_row * n);
        for (int p = 1; p < size; ++p) {
            int p_start = p * rows_per_proc + (p < remainder ? p : remainder);
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            Matrix temp(p_rows * n);
            MPI_Recv(temp.data(), p_rows * n, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::copy(temp.begin(), temp.end(), C.begin() + p_start * n);
        }
        if (writeMatrix(fileC, C, n)) {
            std::cout << "Результат записан в " << fileC << std::endl;
        } else {
            std::cerr << "Ошибка записи результата" << std::endl;
        }
    } else {
        MPI_Send(C_local.data(), local_rows * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
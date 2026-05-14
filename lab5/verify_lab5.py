#!/usr/bin/env python3

import subprocess
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCS = [1, 2, 4, 8]
MATRIX1_TEMPLATE = "matrix1_{}.txt"
MATRIX2_TEMPLATE = "matrix2_{}.txt"
RESULT_TEMPLATE = "result_{}.txt"
EXECUTABLE = "mpirun -np {procs} --oversubscribe --use-hwthread-cpus ./lab3"
CSV_FILENAME = "results_mpi.csv"
PLOT_FILENAME = "time_vs_size_mpi.png"

def generate_matrix(n, filename):
    matrix = np.random.uniform(-10.0, 10.0, (n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(f"{x:.10f}" for x in row) + "\n")
    print(f"  Сгенерирована матрица {n}x{n} в {filename}")

def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        data = []
        for _ in range(n):
            row = list(map(float, f.readline().split()))
            if len(row) != n:
                raise ValueError(f"Неверный формат строки в {filename}")
            data.append(row)
    return np.array(data)

def verify_result(n, fileA, fileB, fileC):
    try:
        A = read_matrix(fileA)
        B = read_matrix(fileB)
        C_prog = read_matrix(fileC)
        C_ref = np.dot(A, B)
        if np.allclose(C_prog, C_ref, rtol=1e-9, atol=1e-9):
            return True, None
        else:
            diff = np.max(np.abs(C_prog - C_ref))
            return False, diff
    except Exception as e:
        return False, str(e)

def run_benchmark():
    if not os.path.isfile("./lab3"):
        print("Ошибка: исполняемый файл './lab3' не найден.")
        print("Скомпилируйте программу: mpicxx -std=c++17 -O2 lab3.cpp -o lab3")
        sys.exit(1)

    results = []

    print("Начинаем бенчмарк для размеров:", SIZES)
    print("Количество процессов:", PROCS)

    for n in SIZES:
        print(f"\n=== Размер {n} ===")
        f1 = MATRIX1_TEMPLATE.format(n)
        f2 = MATRIX2_TEMPLATE.format(n)
        fres = RESULT_TEMPLATE.format(n)

        generate_matrix(n, f1)
        generate_matrix(n, f2)

        for p in PROCS:
            print(f"  Запуск с {p} процесс(ами)...")
            cmd = EXECUTABLE.format(procs=p).split() + [f1, f2, fres]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            except subprocess.TimeoutExpired:
                print(f"    Превышено время ожидания для размера {n}, процессов {p}")
                results.append((n, p, None, "Timeout"))
                continue
            except Exception as e:
                print(f"    Ошибка запуска: {e}")
                results.append((n, p, None, f"Error: {e}"))
                continue

            if result.returncode != 0:
                print(f"    Программа завершилась с ошибкой (код {result.returncode})")
                print("    STDERR:", result.stderr)
                results.append((n, p, None, "Runtime error"))
                continue

            time_line = None
            for line in result.stdout.split('\n'):
                if "Время умножения" in line:
                    time_line = line
                    break
            if time_line:
                try:
                    parts = time_line.split(':')[1].strip().split()
                    mult_time = float(parts[0])
                except:
                    mult_time = None
            else:
                mult_time = None

            if mult_time is None:
                print("    Не удалось извлечь время умножения")
                results.append((n, p, None, "Parse error"))
                continue

            ok, info = verify_result(n, f1, f2, fres)
            status = "OK" if ok else f"FAIL ({info})"
            print(f"    Время: {mult_time:.6f} с, Статус: {status}")
            results.append((n, p, mult_time, status))

    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "Processes", "Time (s)", "Status"])
        for n, p, tm, st in results:
            if tm is not None:
                writer.writerow([n, p, f"{tm:.6f}", st])
            else:
                writer.writerow([n, p, "N/A", st])

    print(f"\nРезультаты сохранены в {CSV_FILENAME}")

    plt.figure(figsize=(10, 6))
    for p in PROCS:
        valid = [(n, tm) for (n, proc, tm, st) in results if proc == p and tm is not None]
        if not valid:
            continue
        sizes, times = zip(*sorted(valid, key=lambda x: x[0]))
        plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, label=f'Процессов: {p}')
    plt.xlabel('Размер матрицы (n)')
    plt.ylabel('Время (с)')
    plt.title('Зависимость времени умножения от размера и числа процессов MPI')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILENAME)
    print(f"График сохранён как {PLOT_FILENAME}")

    print("\n## Таблица результатов (время в секундах)\n")
    sizes_sorted = sorted(set([n for (n, _, _, _) in results]))
    procs_sorted = sorted(set([p for (_, p, _, _) in results]))
    header = "| Размер | " + " | ".join(f"{p}" for p in procs_sorted) + " |"
    print(header)
    print("|--------|" + "|".join(["---------"] * len(procs_sorted)) + "|")
    for n in sizes_sorted:
        row = f"| {n} |"
        for p in procs_sorted:
            tm = next((tm for (sz, proc, tm, _) in results if sz == n and proc == p and tm is not None), None)
            if tm is not None:
                row += f" {tm:.6f} |"
            else:
                row += " N/A |"
        print(row)

if __name__ == "__main__":
    run_benchmark()
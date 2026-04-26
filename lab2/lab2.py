#!/usr/bin/env python3

import subprocess
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS = [1, 2, 4, 8]
MATRIX1_TEMPLATE = "matrix1_{}.txt"
MATRIX2_TEMPLATE = "matrix2_{}.txt"
RESULT_TEMPLATE = "result_{}.txt"
EXECUTABLE = "./lab2"
CSV_FILENAME = "results_omp.csv"
PLOT_FILENAME = "time_vs_size_threads.png"

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
    if not os.path.isfile(EXECUTABLE):
        print(f"Ошибка: исполняемый файл '{EXECUTABLE}' не найден.")
        print("Скомпилируйте программу: g++ -std=c++17 -O2 -fopenmp lab2.cpp -o lab2")
        sys.exit(1)

    results = []

    print("Начинаем бенчмарк для размеров:", SIZES)
    print("Количество потоков:", THREADS)

    for n in SIZES:
        print(f"\n=== Размер {n} ===")
        f1 = MATRIX1_TEMPLATE.format(n)
        f2 = MATRIX2_TEMPLATE.format(n)
        fres = RESULT_TEMPLATE.format(n)

        generate_matrix(n, f1)
        generate_matrix(n, f2)

        for t in THREADS:
            print(f"  Запуск с {t} потоком(ами)...")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(t)

            try:
                result = subprocess.run(
                    [EXECUTABLE, f1, f2, fres],
                    capture_output=True,
                    text=True,
                    timeout=7200,
                    env=env
                )
            except subprocess.TimeoutExpired:
                print(f"    Превышено время ожидания для размера {n}, потоков {t}")
                results.append((n, t, None, "Timeout"))
                continue
            except Exception as e:
                print(f"    Ошибка запуска: {e}")
                results.append((n, t, None, f"Error: {e}"))
                continue

            if result.returncode != 0:
                print(f"    Программа завершилась с ошибкой (код {result.returncode})")
                print("    STDERR:", result.stderr)
                results.append((n, t, None, "Runtime error"))
                continue

            time_line = None
            for line in result.stdout.split('\n'):
                if "Время умножения:" in line:
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
                results.append((n, t, None, "Parse error"))
                continue

            ok, info = verify_result(n, f1, f2, fres)
            status = "OK" if ok else f"FAIL ({info})"
            if not ok:
                print(f"    Верификация не пройдена: {info}")

            print(f"    Время: {mult_time:.6f} с, Статус: {status}")
            results.append((n, t, mult_time, status))

    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "Threads", "Time (s)", "Status"])
        for n, t, tm, st in results:
            if tm is not None:
                writer.writerow([n, t, f"{tm:.6f}", st])
            else:
                writer.writerow([n, t, "N/A", st])

    print(f"\nРезультаты сохранены в {CSV_FILENAME}")

    plt.figure(figsize=(10, 6))
    for t in THREADS:
        valid = [(n, tm) for (n, thr, tm, st) in results if thr == t and tm is not None]
        if not valid:
            continue
        sizes, times = zip(*sorted(valid, key=lambda x: x[0]))
        plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, label=f'Потоков: {t}')

    plt.xlabel('Размер матрицы (n)')
    plt.ylabel('Время (с)')
    plt.title('Зависимость времени умножения от размера матрицы и числа потоков')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILENAME)
    print(f"График сохранён как {PLOT_FILENAME}")
    plt.show()

    print("\n## Таблица результатов (время в секундах)\n")
    sizes_sorted = sorted(set([n for (n, _, _, _) in results]))
    threads_sorted = sorted(set([t for (_, t, _, _) in results]))
    header = "| Размер | " + " | ".join(f"{t}" for t in threads_sorted) + " |"
    print(header)
    separator = "|--------|" + "|".join(["---------"] * len(threads_sorted)) + "|"
    print(separator)
    for n in sizes_sorted:
        row = f"| {n} |"
        for t in threads_sorted:
            tm = next((tm for (sz, thr, tm, _) in results if sz == n and thr == t and tm is not None), None)
            if tm is not None:
                row += f" {tm:.6f} |"
            else:
                row += " N/A |"
        print(row)

if __name__ == "__main__":
    run_benchmark()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import numpy as np
import os
import sys
import time as pytime
import csv
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]         
MATRIX1_TEMPLATE = "matrix1_{}.txt"
MATRIX2_TEMPLATE = "matrix2_{}.txt"
RESULT_TEMPLATE = "result_{}.txt"
EXECUTABLE = "./matrix_mult" if os.name != "nt" else "matrix_mult.exe"
CSV_FILENAME = "results.csv"                         
PLOT_FILENAME = "time_plot.png"                      

def generate_matrix(n, filename):
    """Генерирует случайную квадратную матрицу n x n и сохраняет в файл"""
    matrix = np.random.uniform(-10.0, 10.0, (n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(f"{x:.10f}" for x in row) + "\n")
    print(f"  Сгенерирована матрица {n}x{n} в {filename}")

def read_matrix(filename):
    """Читает матрицу из файла (формат: размер, затем построчно числа)"""
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
    """Проверяет результат умножения с помощью numpy.dot"""
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
    # Проверка наличия исполняемого файла
    if not os.path.isfile(EXECUTABLE):
        print(f"Ошибка: исполняемый файл '{EXECUTABLE}' не найден.")
        print("Скомпилируйте matrix_mult.cpp перед запуском:")
        print("  g++ -std=c++17 -O2 matrix_mult.cpp -o matrix_mult")
        sys.exit(1)

    results = []  

    print("Начинаем бенчмарк для размеров:", SIZES)
    for n in SIZES:
        print(f"\n=== Размер {n} ===")
        f1 = MATRIX1_TEMPLATE.format(n)
        f2 = MATRIX2_TEMPLATE.format(n)
        fres = RESULT_TEMPLATE.format(n)

        generate_matrix(n, f1)
        generate_matrix(n, f2)

        print(f"  Запуск {EXECUTABLE} {f1} {f2} {fres}")
        try:
            result = subprocess.run(
                [EXECUTABLE, f1, f2, fres],
                capture_output=True,
                text=True,
                timeout=600  
            )
        except subprocess.TimeoutExpired:
            print("  Превышено время ожидания (2 часа)")
            results.append((n, None))
            continue
        except Exception as e:
            print(f"  Ошибка запуска: {e}")
            results.append((n, None))
            continue

        if result.returncode != 0:
            print(f"  Программа завершилась с ошибкой (код {result.returncode})")
            print("  STDERR:", result.stderr)
            results.append((n, None))
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
            except Exception:
                mult_time = None
        else:
            mult_time = None

        if mult_time is None:
            print("  Не удалось извлечь время умножения, пропускаем")
            results.append((n, None))
            continue

        ok, info = verify_result(n, f1, f2, fres)
        if ok:
            status = "OK"
        else:
            status = f"FAIL ({info})"
            print(f"  Верификация не пройдена: {info}")
          
        print(f"  Время умножения: {mult_time:.6f} с, Верификация: {status}")
        results.append((n, mult_time))

    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "Time (s)"])
        for n, t in results:
            if t is not None:
                writer.writerow([n, f"{t:.6f}"])
            else:
                writer.writerow([n, "N/A"])

    print(f"\nРезультаты сохранены в {CSV_FILENAME}")

    valid_results = [(n, t) for n, t in results if t is not None]
    if valid_results:
        sizes, times = zip(*valid_results)
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Размер матрицы (n)')
        plt.ylabel('Время (с)')
        plt.title('Зависимость времени умножения от размера матрицы')
        plt.grid(True)
        plt.savefig(PLOT_FILENAME)
        print(f"График сохранён как {PLOT_FILENAME}")
    else:
        print("Нет данных для построения графика.")

    print("\n## Таблица результатов\n")
    print("| Размер | Время (с) |")
    print("|--------|-----------|")
    for n, t in results:
        if t is not None:
            print(f"| {n} | {t:.6f} |")
        else:
            print(f"| {n} | N/A |")

if __name__ == "__main__":
    run_benchmark()
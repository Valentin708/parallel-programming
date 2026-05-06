#!/usr/bin/env python3

import subprocess
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
BLOCK_SIZES = [8, 16, 32]
MATRIX1_TEMPLATE = "matrix1_{}.txt"
MATRIX2_TEMPLATE = "matrix2_{}.txt"
RESULT_TEMPLATE = "result_{}.txt"
EXECUTABLE = "./lab4"
CSV_FILENAME = "results_cuda.csv"
PLOT_FILENAME = "time_vs_size_cuda.png"

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
        print("Скомпилируйте программу: nvcc -O2 lab4.cu -o lab4")
        sys.exit(1)

    results = []
    print("Начинаем бенчмарк для размеров:", SIZES)
    print("Размеры блоков (threads):", BLOCK_SIZES)

    for n in SIZES:
        print(f"\n=== Размер {n} ===")
        f1 = MATRIX1_TEMPLATE.format(n)
        f2 = MATRIX2_TEMPLATE.format(n)
        fres = RESULT_TEMPLATE.format(n)

        generate_matrix(n, f1)
        generate_matrix(n, f2)

        for bs in BLOCK_SIZES:
            print(f"  Запуск с block_size = {bs} ...")
            cmd = [EXECUTABLE, f1, f2, fres, str(bs)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            except subprocess.TimeoutExpired:
                print(f"    Превышено время ожидания для размера {n}, block_size {bs}")
                results.append((n, bs, None, "Timeout"))
                continue
            except Exception as e:
                print(f"    Ошибка запуска: {e}")
                results.append((n, bs, None, f"Error: {e}"))
                continue

            if result.returncode != 0:
                print(f"    Программа завершилась с ошибкой (код {result.returncode})")
                print("    STDERR:", result.stderr)
                results.append((n, bs, None, "Runtime error"))
                continue

            time_line = None
            for line in result.stdout.split('\n'):
                if "Время умножения на GPU" in line:
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
                results.append((n, bs, None, "Parse error"))
                continue

            ok, info = verify_result(n, f1, f2, fres)
            status = "OK" if ok else f"FAIL ({info})"
            print(f"    Время: {mult_time:.6f} с, Статус: {status}")
            results.append((n, bs, mult_time, status))

    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "BlockSize", "Time (s)", "Status"])
        for n, bs, tm, st in results:
            if tm is not None:
                writer.writerow([n, bs, f"{tm:.6f}", st])
            else:
                writer.writerow([n, bs, "N/A", st])

    print(f"\nРезультаты сохранены в {CSV_FILENAME}")

    plt.figure(figsize=(10, 6))
    for bs in BLOCK_SIZES:
        valid = [(n, tm) for (n, b, tm, st) in results if b == bs and tm is not None]
        if not valid:
            continue
        sizes, times = zip(*sorted(valid, key=lambda x: x[0]))
        plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, label=f'block = {bs}')
    plt.xlabel('Размер матрицы (n)')
    plt.ylabel('Время (с)')
    plt.title('Зависимость времени умножения на GPU от размера и конфигурации блоков')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILENAME)
    print(f"График сохранён как {PLOT_FILENAME}")

    print("\n## Таблица результатов (время в секундах)\n")
    sizes_sorted = sorted(set([n for (n, _, _, _) in results]))
    blocks_sorted = sorted(set([bs for (_, bs, _, _) in results]))
    header = "| Размер | " + " | ".join(f"{bs}" for bs in blocks_sorted) + " |"
    print(header)
    print("|--------|" + "|".join(["---------"] * len(blocks_sorted)) + "|")
    for n in sizes_sorted:
        row = f"| {n} |"
        for bs in blocks_sorted:
            tm = next((tm for (sz, b, tm, _) in results if sz == n and b == bs and tm is not None), None)
            if tm is not None:
                row += f" {tm:.6f} |"
            else:
                row += " N/A |"
        print(row)

if __name__ == "__main__":
    run_benchmark()
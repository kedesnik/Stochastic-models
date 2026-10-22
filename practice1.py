import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import subprocess
import sys
from pathlib import Path
from itertools import product
import os

# Функции для 1 задания
def count_differences(state1, state2):
    return sum(
        a != b
        for a, b in zip(state1, state2)
    )

def graph(states, edges, filename, directed=True):
    # Создаём граф
    g = nx.DiGraph() if directed else nx.Graph()

    # Добавляем состояния
    g.add_nodes_from(states)

    # Добавляем переходы
    for a, b, w in edges:
        g.add_edge(a, b, weight=w)

    # Создаём окно графика
    plt.figure(figsize=(7, 5))

    # Располагаем вершины по окружности
    pos = nx.circular_layout(g)

    # Рисуем граф
    nx.draw_networkx(
        g,
        pos,
        node_size=1800,
        node_color='#cfe7fa',
        arrows=directed,
        arrowsize=22,
        connectionstyle='arc3,rad=0.12'
    )

    # Подписи интенсивностей переходов
    labels = {
        (a, b): str(d['weight'])
        for a, b, d in g.edges(data=True)
        if d['weight'] is not None
    }

    if labels:
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=labels,
            label_pos=0.35
        )

    # Убираем оси координат
    plt.axis('off')
    plt.tight_layout()

    # Сохраняем график
    plt.savefig(filename, dpi=160)
    plt.close()

    print(f"Граф сохранён: {filename}")

    file_path = Path(filename).resolve()
    subprocess.Popen(["xdg-open", str(file_path)])

# Функции для 2 задания
def generate_lamp_transitions(states):
    transitions = []

    for start in states:
        for end in states:
            if start == end:
                continue

            # Из неисправного состояния переходов нет
            if start == "S2":
                continue

            # Между исправными состояниями
            if start in ("S0", "S1") and end in ("S0", "S1"):
                transitions.append((start, end, None))

            # Переход к неисправному состоянию
            elif end == "S2":
                transitions.append((start, end, None))

    return transitions

# # Функции для 3 и 4 задания
def poisson(m, lt):
    return math.exp(-lt) * lt**m / math.factorial(m)


def stationary_ctmc(n, edges):
    """Стационарное распределение непрерывной цепи Маркова; ребро (i,j,rate)."""
    q = np.zeros((n, n))
    for i, j, rate in edges:
        if i == j or not 0 <= i < n or not 0 <= j < n or rate < 0:
            raise ValueError('Неверное ребро')
        q[i, j] += rate
    np.fill_diagonal(q, -q.sum(axis=1))
    a = q.T.copy(); a[-1, :] = 1
    b = np.zeros(n); b[-1] = 1
    p = np.linalg.solve(a, b)
    return p





def loss_system(lam, mu, channels):
    """M/M/c/c (Эрланг B), интенсивности на одинаковую единицу времени."""
    a = lam / mu
    weights = [1.0]
    for j in range(1, channels + 1): weights.append(weights[-1] * a / j)
    z = sum(weights); p = [x / z for x in weights]
    blocking = p[-1]; q = 1 - blocking; throughput = lam * q
    return dict(p=p, P_refusal=blocking, Q=q, A=throughput,
                busy=throughput / mu, idle=channels - throughput / mu)


def finite_queue(lam, mu, channels, waiting):
    """M/M/c/(c+m), в том числе M/M/1/K."""
    capacity = channels + waiting
    w = [1.0]
    for n in range(1, capacity + 1):
        w.append(w[-1] * lam / (min(n, channels) * mu))
    z = sum(w); p = [x / z for x in w]
    q = 1 - p[-1]; throughput = lam * q
    Lq = sum(max(n - channels, 0) * p[n] for n in range(capacity + 1))
    L = sum(n * p[n] for n in range(capacity + 1))
    busy = sum(min(n, channels) * p[n] for n in range(capacity + 1))
    return dict(p=p, P_refusal=p[-1], Q=q, A=throughput,
                busy=busy, Lq=Lq, L=L, Wq=Lq / throughput, W=L / throughput)


def report(result):
    for k, v in result.items():
        if isinstance(v, list): print(f'  {k}: ' + ', '.join(f'{x:.6f}' for x in v))
        else: print(f'  {k}: {v:.6f}')


def graph_from_user(task, n):
    """Для схем из DOCX направления стрелок не читаются надёжно: вводятся вручную."""
    print(f'Задание {task}: введите ориентированные переходы «от куда интенсивность».')
    print('Нумерация состояний: 1..' + str(n) + '. Пустая строка завершает ввод.')
    edges = []
    while True:
        line = input('Ребро (например 1 2 3): ').strip()
        if not line: break
        a, b, rate = line.split()
        edges.append((int(a) - 1, int(b) - 1, float(rate)))
    if not edges:
        print('Нет ребер: задача пропущена, чтобы не придумывать направления стрелок.')
        return
    graph([f'S{i}' for i in range(1, n + 1)],
          [(f'S{i+1}', f'S{j+1}', r) for i, j, r in edges], f'task_{task}.png')
    try: print('Предельные вероятности:', stationary_ctmc(n, edges))
    except np.linalg.LinAlgError: print('Нет единственного решения; проверьте граф и замкнутые классы.')

def generate_states(n):
    return [
        "".join(bits)
        for bits in product("01", repeat=n)
    ]

def generate_transitions(states):
    transitions = []

    for state1 in states:
        for state2 in states:

            if count_differences(state1, state2) == 1:
                transitions.append(
                    (state1, state2, None)
                )

    return transitions

def task1():
    states = generate_states(2)
    transitions = generate_transitions(states)

    print("Состояния:", states)
    print("Переходы:")

    for start, end, _ in transitions:
        print(f"{start} → {end}")

    graph(states, transitions, "task_1.png")

def task2():
    states = ["S0", "S1", "S2"]

    transitions = generate_lamp_transitions(states)

    print("\nСостояния лампочки:")
    print("S0 — выкл")
    print("S1 — вкл")
    print("S2 — слом")

    print("\nВозможные переходы:")
    for start, end, _ in transitions:
        print(f"{start} → {end}")

    graph(states, transitions, "task_2.png")


def task3():
    lam = 3
    time = 2
    lt = lam * time
    print('а) P(N=4):', poisson(4,lt), '\nб) P(N>=1):', 1-math.exp(-lt), '\nв) P(N=0):', math.exp(-lt))

def task4():
    m = (2 / 5) * 15
    print('а) P(N=11):', poisson(11,m), 'б) P(N>=1):', 1-math.exp(-m), 'в) P(N=0):', math.exp(-m))


def task6():
    # Состояние i = число чёрных шаров в первой урне, i=0,1,2.
    p = np.zeros((3,3))
    for i in range(3):
        if i >= 1: p[i,i-1] = (i / 2)**2
        if i <= 1: p[i,i+1] = ((2-i)/2)**2
        p[i,i] = 1 - sum(p[i])
    a = p.T - np.eye(3); a[-1,:] = 1
    b = np.array([0.,0.,1.]); pi = np.linalg.solve(a,b)
    print('Матрица переходов:\n',p,'\nПредельные вероятности:',pi)
    graph(['0','1','2'], [('0','1',1),('1','0',0.25),('1','2',0.25),('2','1',1),('1','1',0.5)],'task_6.png')


def task9(): report(loss_system(36/24, 1/0.5, 1))
def task10():
    r = loss_system(.4, 1/3, 1); report(r)
    print('Номинальная пропускная способность при непрерывной загрузке: 1/3 вызова/мин')
def task11():
    report(loss_system(36/24,2,4))
    c=1
    while loss_system(36/24,2,c)['Q'] < .9: c+=1
    print('Минимум каналов при Q >= 0.9:',c)
def task12():
    for c in (2,3):
        r=loss_system(4,1/.8,c)
        profit=4*r['A'] - 2*c
        print(f'{c} каналов: A={r["A"]:.6f} заявок/ч, прибыль={profit:.6f} ден.ед./ч')
def task13(): report(finite_queue(5,6,1,1))
def task14(): report(finite_queue(.85,1/1.05,1,3))
def task15():
    # Предположение: 4 заявки/час — интенсивность каждого канала.
    report(finite_queue(2,4,2,2))
def task16(): report(finite_queue(12,3,3,3))



def main():
    tasks = {
        1: task1,
        2: task2,
        3: task3,
        4: task4,
        #5:task5,
        6: task6,
        #7: task7,
        #8: task8,
        9: task9,
        10: task10,
        11: task11,
        12: task12,
        13: task13,
        14: task14,
        15: task15,
        16: task16,
    }

    while True:
        print("Выберите задание от 1 до 16")
        print("0 — выход")

        choice = input("Ваш выбор: ").strip()

        if choice == "0":
            print("Завершение программы.")
            break

        if not choice.isdigit():
            print("Ошибка: введите номер задания.")
            continue

        number = int(choice)

        if number not in tasks:
            print("Такого задания нет.")
            continue

        print(f"\n=== ЗАДАНИЕ {number} ===")

        try:
            tasks[number]()
        except Exception as error:
            print(f"Ошибка выполнения: {error}")


if __name__ == "__main__":
    main()

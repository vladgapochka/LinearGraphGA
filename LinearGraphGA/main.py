import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLineEdit, QLabel, QFormLayout, QTextEdit)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx
import random
import numpy as np
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Graph Placement Genetic Algorithm')
        self.setGeometry(100, 100, 1200, 900)  # Adjusted size for better fitting

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Horizontal layout for inputs and console output
        top_layout = QHBoxLayout()
        form_layout = QFormLayout()
        self.vertices_input = QLineEdit("12")
        self.prob_input = QLineEdit("0.5")
        self.pop_size_input = QLineEdit("100")
        self.gen_input = QLineEdit("50")
        self.mut_rate_input = QLineEdit("0.1")

        form_layout.addRow('Кол-во Вершин:', self.vertices_input)
        form_layout.addRow('Вероятность возникновения ребра:', self.prob_input)
        form_layout.addRow('Размер популяции:', self.pop_size_input)
        form_layout.addRow('Кол-во поколений:', self.gen_input)
        form_layout.addRow('Частота мутаций:', self.mut_rate_input)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedWidth(400)

        top_layout.addLayout(form_layout)
        top_layout.addWidget(self.log)
        main_layout.addLayout(top_layout)

        run_button = QPushButton('Запустить ГА', self)
        run_button.clicked.connect(self.run_algorithm)
        main_layout.addWidget(run_button)

        # Horizontal layout for best solution graph
        middle_layout = QHBoxLayout()
        self.canvas2 = FigureCanvas(plt.figure(figsize=(6, 4)))
        middle_layout.addWidget(self.canvas2)
        main_layout.addLayout(middle_layout)

        # Horizontal layout for initial graph and fitness history graph
        graph_layout = QHBoxLayout()
        self.canvas1 = FigureCanvas(plt.figure(figsize=(6, 4)))
        graph_layout.addWidget(self.canvas1)

        self.canvas3 = FigureCanvas(plt.figure(figsize=(6, 4)))
        graph_layout.addWidget(self.canvas3)

        main_layout.addLayout(graph_layout)

    def run_algorithm(self):
        num_vertices = int(self.vertices_input.text())
        prob = float(self.prob_input.text())
        pop_size = int(self.pop_size_input.text())
        generations = int(self.gen_input.text())
        mutation_rate = float(self.mut_rate_input.text())

        # Засекаем время начала выполнения алгоритма
        start_time = time.time()

        G = nx.gnp_random_graph(num_vertices, prob)
        self.plot_graph(G, self.canvas1.figure, "Начальный Граф")
        best_solution, best_fitness, fitness_history = self.genetic_algorithm(G, pop_size, generations, mutation_rate)

        # Выводим результаты и время выполнения
        elapsed_time = time.time() - start_time
        self.plot_best_solution(G, best_solution, self.canvas2.figure, "Размещение вершин графа на линейке")
        self.plot_fitness_history(fitness_history, self.canvas3.figure,
                                  "График зависимости лучших решений от поколений")
        self.log.append(f"Наименьшее растояние: {best_fitness}, Лучшее решение: {best_solution}")
        self.log.append(f"Время выполнения алгоритма: {elapsed_time:.2f} секунд")

    def plot_graph(self, G, figure, title):
        figure.clf()
        ax = figure.add_subplot(111)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True)
        ax.set_title(title)
        self.canvas1.draw()

    def plot_best_solution(self, G, solution, figure, title):
        figure.clf()
        ax = figure.add_subplot(111)
        pos = {node: (i, 0) for i, node in enumerate(solution)}
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='orange')
        ax.set_title(title)
        self.canvas2.draw()

    def plot_fitness_history(self, fitness_history, figure, title):
        figure.clf()
        ax = figure.add_subplot(111)
        ax.plot(fitness_history, label='Лучшее решение')
        ax.set_xlabel('Поколения')
        ax.set_ylabel('Лучшее растояние')
        ax.set_title(title)
        ax.legend()
        self.canvas3.draw()

    def genetic_algorithm(self, G, population_size, generations, mutation_rate):
        def initialize_population(size, num_vertices):
            return [random.sample(range(num_vertices), num_vertices) for _ in range(size)]

        def calculate_fitness(solution):
            total_distance = 0
            for u, v in G.edges:
                total_distance += abs(solution.index(u) - solution.index(v))
            return total_distance

        population = initialize_population(population_size, len(G.nodes))
        best_fitness = float('inf')
        best_solution = None
        fitness_history = []

        for generation in range(generations):
            fitness_scores = [calculate_fitness(individual) for individual in population]
            best_current_idx = np.argmin(fitness_scores)
            best_current_fitness = fitness_scores[best_current_idx]
            if best_current_fitness < best_fitness:
                best_fitness = best_current_fitness
                best_solution = population[best_current_idx]

            fitness_history.append(best_fitness)
            print(f"Поколение {generation}: Наилучшее расстояние = {best_fitness}, Лучшее решение = {best_solution}")

            new_population = []
            for _ in range(population_size // 2):
                parent1, parent2 = random.sample(population, 2)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
            population = new_population

        return best_solution, best_fitness, fitness_history


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
    return child

def mutate(solution, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(solution)), 2)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

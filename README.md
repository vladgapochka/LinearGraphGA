# LinearGraphGA

LinearGraphGA - это программа, которая использует генетический алгоритм для решения задачи размещения вершин графа на линейке. Этот проект позволяет пользователю исследовать эффективные способы размещения вершин графа на линейной структуре, оптимизируя расстояния между вершинами.

## Описание

Данная программа позволяет пользователю задать параметры графа и генетического алгоритма, а затем запустить алгоритм для нахождения оптимального размещения вершин графа на линейке. После завершения выполнения алгоритма, программа выводит наилучшее найденное решение, а также визуализирует начальный граф, найденное размещение и график прогресса оптимизации.

## Функциональности

- **Генетический алгоритм**: Реализация генетического алгоритма для оптимизации размещения вершин графа на линейке.
- **Визуализация**: Возможность визуализировать начальный граф, оптимальное размещение вершин и график прогресса оптимизации.
- **Параметры настройки**: Возможность задать параметры графа (количество вершин, вероятность появления ребра) и параметры генетического алгоритма (размер популяции, количество поколений, частота мутаций).

## Использование

1. Установите необходимые зависимости, запустив `pip install -r requirements.txt`.
2. Запустите `main.py`, чтобы открыть приложение.
3. Введите параметры графа и генетического алгоритма.
4. Нажмите кнопку "Запустить ГА", чтобы запустить генетический алгоритм.
5. Просмотрите результаты на графиках и в логе.

## Требования к окружению

- Python 3.12
- PyQt5
- NetworkX
- Matplotlib
- NumPy

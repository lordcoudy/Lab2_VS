#include <stdcpp.h>
#include <omp.h>

#pragma region Определения
// Определение индекса элемента в матрице
#define index(x,y) (x + y * size)
// Хардкод количества процессов для 2 задания
#define N 8
// Шаблоны для подсчета времени выполнения
#define countTime double time2 = omp_get_wtime(); double time = time2 - time1;
#define write ompTimes << thread << "," << size << "," << time << endl;
#pragma endregion

using namespace std;
using namespace std::chrono;

// Объявление указателей для матриц
double *A, *B, *C, *Y;

#pragma region Функции с массивами

// Функция для расчёта времени выполнения функции
template<typename Func>
double measureExecutionTime(Func func) {
    auto startTime = high_resolution_clock::now();
    func();
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(endTime - startTime).count();
    return duration / 1e6;                      // Перевод в миллисекунды
}

// Функция для создания матрицы
double* createMatrix(int size) {
    auto* matrix = new double[size*size];       // Динамическая инициализация матрицы
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[index(i,j)] = i + j + 1.5;
        }
    }
    return matrix;
}

// Процедура для удаления матрицы
void deleteMatrix(const double* matrix, int size) {
    delete[] matrix;
}

// Процедура для инициализации всех массивов
void init (const int & size)
{
    A = createMatrix(size);
    B = createMatrix(size);
    C = createMatrix(size);
    Y = new double[size*size];
}

// Процедура для удаления всех массивов
void finalize (const int & size)
{
    deleteMatrix(A, size);
    deleteMatrix(B, size);
    deleteMatrix(C, size);
    deleteMatrix(Y, size);
}

#pragma endregion

#pragma region OpenMP подход 1
// Процедура для выполнения вычислений с использованием OpenMP, подход 1
void runVar1(const int & thread, const int & size)
{
    init(size);                                 // Инициализация массивов
    int i, j, rank, pSize;
    double time1 = omp_get_wtime();             // Замер времени начала работы

    // Параллельная секция с приватными переменными
#pragma omp parallel private (i, j, rank, pSize)
    {
        rank = omp_get_thread_num();            // Получение номера потока
        pSize = omp_get_num_threads();          // Получение общего количества потоков

        // Расчет элементов матрицы Y в соответствии с номером потока
        for (i = size * rank / pSize; i < size * (rank + 1) / pSize; i++) {
            for (j = 0; j < size; ++j) {
                Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
            }
        }
    }
    countTime                                   // Подсчет времени выполнения
    ofstream ompTimes (
            "./ompTimes1.csv", ios::app);       // Открытие файла для записи времени выполнения
    write                                       // Запись времени выполнения в файл
    finalize(size);                             // Удаление всех массивов
}
#pragma endregion

#pragma region OpenMP подход 2
// Процедура для выполнения вычислений с использованием OpenMP, подход 2
void runVar2(const int & thread, const int & size)
{
    init(size);                                 // Инициализация массивов
    double time1 = omp_get_wtime();             // Замер времени начала работы

    // Параллельная секция с использованием секций
#pragma omp parallel sections
    {
    // Секция с вычислением части массива
#pragma omp section
        {
            for (int i = 0; i < size / N; i++) {
                for (int j = 0; j < size; ++j) {
                    Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
                }
            }
        }
#pragma omp section
        {
            for (int i = size / N; i < 2 * size / N; i++) {
                for (int j = 0; j < size; ++j) {
                    Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
                }
            }
        }
#pragma omp section
        {
            for (int i = 2 * size / N; i < 3 * size / N; i++) {
                for (int j = 0; j < size; ++j) {
                    Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
                }
            }
        }
#pragma omp section
        {
            for (int i = 3 * size / N; i < 4 * size / N; i++) {
                for (int j = 0; j < size; ++j) {
                    Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
                }
            }
        }
#pragma omp section
        {
            for (int i = 4 * size / N; i < 5 * size / N; i++) {
                for (int j = 0; j < size; ++j) {
                    Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
                }
            }
        }
#pragma omp section
        {
            for (int i = 5 * size / N; i < 6 * size / N; i++) {
                for (int j = 0; j < size; ++j) {
                    Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
                }
            }
        }
#pragma omp section
        {
            for (int i = 6 * size / N; i < 7 * size / N; i++) {
                for (int j = 0; j < size; ++j) {
                    Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
                }
            }
        }
#pragma omp section
        {
            for (int i = 7 * size / N; i < 8 * size / N; i++) {
                for (int j = 0; j < size; ++j) {
                    Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
                }
            }
        }
    }
    countTime                                   // Подсчет времени выполнения
    ofstream ompTimes (
            "./ompTimes2.csv", ios::app);       // Открытие файла для записи времени выполнения
    write                                       // Запись времени выполнения в файл
    finalize(size);                             // Удаление всех массивов
}
#pragma endregion

#pragma region OpenMP подход 3
// Процедура для выполнения вычислений с использованием OpenMP, подход 3
void runVar3 (const int & thread, const int & size)
{
    init(size);                                 // Инициализация массивов
    int i, j, rank, pSize;
    double time1 = omp_get_wtime();             // Замер времени начала работы
#pragma omp parallel private (j)
    {
#pragma omp for //schedule(static, size)
        for (i = 0; i < size; i++) {
            for (j = 0; j < size; j++) {
                Y[index(i, j)] = (A[index(i, j)] + C[index(i, j)]) * B[index(i, j)] + A[index(i, j)] / C[index(i, j)];
            }
        }
    }
    countTime                                   // Подсчет времени выполнения
    ofstream ompTimes (
            "./ompTimes3.csv", ios::app);       // Открытие файла для записи времени выполнения
    write                                       // Запись времени выполнения в файл
    finalize(size);                             // Удаление всех массивов
}
#pragma endregion

// Вариант 2, (A + C)*B + A / C
// 1 - 8 threads
// 256, 512, 1024, 2048, 4096

int main() {
#pragma region Инициализация векторов размеров и потоков
    vector<int> sizes =
            {256, 512, 1024, 2048, 4096};       // Размеры массивов
    vector<int> threads = {1, 2, 4, 8};         // Число потоков
#pragma endregion
#pragma region Работа с файлами таблиц
    // Открытие файлов для записи времени выполнения
    ofstream preciseTimes ("./preciseTimes.csv", ios::app);
    ofstream ompTimes1 ("./ompTimes1.csv", ios::app);
    ofstream ompTimes2 ("./ompTimes2.csv", ios::app);
    ofstream ompTimes3 ("./ompTimes3.csv", ios::app);
    // Запись в файлы заголовков
    preciseTimes << "Число потоков" << "," << "Размер" << "," << "Подход 1" << "," << "Подход 2" << "," << "Подход 3" << endl;
    ompTimes1 << "Число потоков" << "," << "Размер" << "," << "Подход 1" << endl;
    ompTimes2 << "Число потоков" << "," << "Размер" << "," << "Подход 2" << endl;
    ompTimes3 << "Число потоков" << "," << "Размер" << "," << "Подход 3" << endl;
#pragma endregion
#pragma region Выполнение вычислений
    for (const auto & thread : threads) {
        omp_set_num_threads(thread);            // Установка числа потоков
        for (const auto &size: sizes) {
            // Замеры времени выполнения в зависимости от подхода, размера массивов и количества потоков
            double time1 = measureExecutionTime([&]() {runVar1(thread, size);});
            double time2 = measureExecutionTime([&]() {runVar2(thread, size);});
            double time3 = measureExecutionTime([&]() {runVar3(thread, size);});
            // Запись в файл результатов
            preciseTimes << thread << "," << size << "," << time1 << "," << time2 << "," << time3 << endl;
        }
    }
#pragma endregion
#pragma region Закрытие файлов
    preciseTimes.close();
    ompTimes1.close();
    ompTimes2.close();
    ompTimes3.close();
#pragma endregion
    return 0;
}

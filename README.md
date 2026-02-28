# Task 17 — Counterexample Finder

Система для поиска контрпримеров к целочисленности LP-релаксации булевой модели задачи (17) из статьи Симанчёва и Уразовой.

Контрпример — это инстанс `(n, d, p, w, r)`, для которого LP-релаксация (переменные `x, y in [0,1]` вместо `{0,1}`) возвращает дробное оптимальное решение.

## Архитектура

```
                  ┌────────────┐
                  │   Oracle   │  генерирует инстанс (n, d, p, w, r)
                  │ random/llm │
                  └─────┬──────┘
                        │ Instance
                        v
                  ┌────────────┐
                  │  Resolver  │  строит LP-модель (Pyomo) и решает (HiGHS)
                  └─────┬──────┘
                        │ SolveResult
                        v
                  ┌────────────┐
                  │  Verifier  │  проверяет: все ли переменные целочисленны?
                  └─────┬──────┘
                        │ VerificationResult
                        v
              ┌─────────────────────┐
              │     Main Loop       │
              │  - собирает Feedback│
              │  - пишет в history  │
              │  - передаёт Oracle  │
              └─────────────────────┘
```

Если Verifier обнаруживает дробные переменные — контрпример найден, цикл останавливается. Иначе Oracle получает историю всех попыток и генерирует следующий инстанс.

## Структура файлов

```
math/
├── model17.py              # Pyomo-модель задачи (17): build_model(), solve_model()
├── search_fractional.py    # Автономный скрипт перебора (random/grid)
│
├── models.py               # Датаклассы: Instance, SolveResult, VerificationResult, Feedback
├── config.py               # Конфигурация: Config (max_iterations, tolerance, ...)
├── resolver.py             # solve(instance) → SolveResult через model17
├── verifier.py             # verify(result) → VerificationResult
├── history.py              # Запись/чтение истории попыток (JSONL)
├── main.py                 # Главный цикл оркестрации
│
├── oracle/
│   ├── __init__.py
│   ├── base.py             # Абстрактный интерфейс OracleBase
│   ├── random_oracle.py    # Случайная генерация инстансов
│   └── llm_oracle.py       # Генерация через LLM (OpenAI API)
│
└── requirements.txt        # pyomo, highspy, openai
```

## Описание файлов

### `models.py`

Четыре датакласса, описывающих данные, проходящие через систему:

| Класс | Поля | Назначение |
|---|---|---|
| `Instance` | `n, d, p, w, r` | Входные параметры задачи |
| `SolveResult` | `status, objective_value, solution_x, solution_y` | Результат решения LP |
| `VerificationResult` | `is_integer, is_counterexample, non_integer_vars` | Результат проверки целочисленности |
| `Feedback` | `iteration, instance, solve_result, verification` | Объединяет всё для одной итерации |

### `model17.py`

Оригинальная Pyomo-модель задачи (17):
- `build_model(inst, integral=True/False)` — строит модель с бинарными или непрерывными переменными
- `solve_model(model, solver_name)` — решает через HiGHS
- Ограничения: (2) назначение слотов, (4) времена выпуска, (14) периодичность, (15) связь x и y, (16) монотонность y

### `resolver.py`

- `solve(instance, solver_name) -> SolveResult`
- Обёртка над `model17.build_model(integral=False)` + `solve_model()`
- Извлекает значения всех переменных `x[i,k]` и `y[i,k]` в словари

### `verifier.py`

- `verify(result, tolerance) -> VerificationResult`
- Проверяет каждую переменную: `|val - round(val)| < tolerance`
- Если хотя бы одна переменная дробная — `is_counterexample = True`

### `oracle/base.py`

Абстрактный базовый класс:
```python
class OracleBase(ABC):
    def generate_initial(self) -> Instance: ...
    def generate_next(self, history: list[Feedback]) -> Instance: ...
```

### `oracle/random_oracle.py`

Генерирует случайные инстансы. Не учитывает историю. Подходит для базового перебора.

### `oracle/llm_oracle.py`

Генерирует инстансы через OpenAI API:
- Отправляет LLM системный промпт с описанием задачи и эвристиками для поиска дробных решений
- В `generate_next()` форматирует историю (статистику + последние 10 попыток) в промпт
- Парсит JSON-ответ, валидирует ограничения (d = n*p, len(w) = n и т.д.)
- При ошибках — до 2 повторов, затем fallback на случайную генерацию

### `history.py`

- `write_feedback(feedback, path)` — дописывает одну строку в JSONL-файл
- `load_history(path) -> list[Feedback]` — читает историю обратно

Формат строки (JSONL):
```json
{"iteration": 0, "timestamp": "2026-02-28T12:00:00+00:00", "instance": {"n": 5, "d": 10, "p": 2, "w": [1,2,3,4,5], "r": [0,1,0,2,1]}, "status": "optimal", "objective": 42.0, "is_integer": true, "is_counterexample": false, "non_integer_vars": {}}
```

### `config.py`

```python
@dataclass
class Config:
    max_iterations: int = 100
    integrality_tolerance: float = 1e-6
    history_file: str = "history.jsonl"
    solver_name: str = "appsi_highs"
```

### `main.py`

Оркестратор. CLI-интерфейс с выбором оракула и параметров запуска. Цикл: oracle → solve → verify → log → repeat.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Использование

### Запуск со случайным оракулом

```bash
python main.py --oracle random --max-iter 500
```

### Запуск с LLM-оракулом

Требует переменную окружения `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY="sk-..."
python main.py --oracle llm --max-iter 50
```

С другой моделью:

```bash
python main.py --oracle llm --model gpt-4o --max-iter 20
```

### Все параметры CLI

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--oracle` | `random` | Тип оракула: `random` или `llm` |
| `--model` | `gpt-4o-mini` | Модель OpenAI (только для `--oracle llm`) |
| `--max-iter` | `100` | Максимальное число итераций |
| `--tol` | `1e-6` | Допуск для проверки целочисленности |
| `--history` | `history.jsonl` | Путь к файлу истории |
| `--solver` | `appsi_highs` | Имя солвера Pyomo |

### Продолжение поиска

История сохраняется в JSONL-файле. При повторном запуске с тем же `--history` нумерация итераций продолжается, а оракул получает всю предыдущую историю:

```bash
python main.py --oracle llm --max-iter 20 --history search1.jsonl
# ... позже:
python main.py --oracle llm --max-iter 20 --history search1.jsonl  # продолжит с итерации 20
```

### Старые скрипты

Автономный перебор (без оракул-системы):

```bash
python model17.py --relax              # решить LP-релаксацию одного инстанса
python search_fractional.py --mode random --n 8 --p 3 --trials 2000
python search_fractional.py --mode grid --n 4 --p 2 --w-values 1,2 --r-values 0,1,2,3
```

## Создание своего оракула

Наследуйте `OracleBase` и реализуйте два метода:

```python
from oracle.base import OracleBase
from models import Instance, Feedback

class MyOracle(OracleBase):
    def generate_initial(self) -> Instance:
        return Instance(n=6, d=12, p=2, w=[1,2,3,4,5,6], r=[0,0,0,0,0,0])

    def generate_next(self, history: list[Feedback]) -> Instance:
        # анализировать history и генерировать следующий инстанс
        ...
```

Затем зарегистрируйте его в `main.py` или используйте напрямую:

```python
from main import run
from config import Config

oracle = MyOracle()
result = run(oracle, Config(max_iterations=50))
```

# NumCompute-Stream

> **A decision tree-based streaming machine learning framework in plain Python, NumPy and matplotlib.**
> Individual extension of the NumCompute group project, built for *Programming for AI (Assignment 2.2)* at Adelaide University.


---

## 1. Project Overview

**NumCompute-Stream** processes data in chunks, updates preprocessing statistics incrementally, and periodically rebuilds a decision tree from buffered data. It includes streaming evaluation, per-chunk logs and reusable matplotlib plots.

The framework uses plain Python, NumPy and matplotlib. It does not depend on pandas, scikit-learn, SciPy or PyTorch. pytest and pytest-cov are development tools only.

### Core Technical Pillars

1. **Buffered Retraining**: `partial_fit()` adds each chunk to a buffer. Once at least `buffer_size` rows have arrived, the tree is rebuilt on the buffered rows and the buffer is cleared. Between retrains, `predict()` uses the most recently trained tree.
2. **Incremental Preprocessing**: Scalers, the imputer and the one-hot encoder extend their learned state with `partial_fit()`. The streaming pipeline updates preprocessing and the tree together, so the active tree always matches its feature transformation.
3. **Streaming Evaluation**: Each chunk is scored before it is used for training (test-then-train). Metrics can be cumulative or rolling, and the confusion matrix and ROC AUC accumulate over time.
4. **Vectorised Split Search**: Gini and entropy split scores are computed from NumPy cumulative class counts, and prediction routes rows together at each node.
5. **Numerical Stability**: NaN-aware statistics, fixed histogram edges across chunks, deterministic tie handling, safe handling of zero-variance chunks and tiny-scale inputs.
6. **Built-in Visualisation**: Three reusable plotting functions for metrics over time, model comparison and predictions versus ground truth.

### Repository & Authorship

- **Individual assignment**: [Legacy164/NumCompute_2, `assignment-2.2` branch](https://github.com/Legacy164/NumCompute_2/tree/assignment-2.2)
- **Assignment history**: [Commits on `assignment-2.2`](https://github.com/Legacy164/NumCompute_2/commits/assignment-2.2)
- **Original group repository**: [DivyViradiya/NumCompute, `main` branch](https://github.com/DivyViradiya/NumCompute/tree/main)
- **Individual student**: Tanmay Patel (`Legacy164`)

The fork's [`main` branch](https://github.com/Legacy164/NumCompute_2/tree/main) preserves the original group project unchanged. All Assignment 2.2 work, including corrections based on Assignment 1 feedback, is on `assignment-2.2`.

---

## 2. Directory Structure

```text
NumCompute_2/
├── numcompute/                     # Inherited group package (CSV reader, batch tools, Assignment 1 fixes)
│   ├── io.py                       # Custom CSV reader with chunked streaming (iter_csv_chunks)
│   ├── preprocessing.py            # Batch scalers, imputer and encoder (base classes)
│   ├── stats.py                    # Descriptive statistics and RunningStats (Welford)
│   ├── metrics.py                  # Classification metrics, confusion matrix and ROC AUC
│   ├── pipeline.py                 # Transformer/Estimator API and batch Pipeline
│   ├── utils.py                    # Validation helpers, NotFittedError, pairwise distances
│   └── ...                         # sort_search, rank, optim, benchmarking
├── numcompute_stream/              # Assignment 2.2 streaming package
│   ├── tree.py                     # Impurity functions, numeric splits and buffered decision tree
│   ├── preprocessing.py            # Incremental StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
│   ├── stats.py                    # StreamStats: streaming moments, exact quantiles and histograms
│   ├── metrics.py                  # StreamingMetrics and StreamingAUC (cumulative and rolling)
│   ├── pipeline.py                 # Synchronised preprocessing and buffered tree updates
│   ├── stream.py                   # StreamTrainer: chunk training, evaluation and logging
│   └── visualise.py                # Three reusable plotting functions
├── tests/                          # Inherited tests plus streaming extension tests (339 total)
│   ├── test_tree.py
│   ├── test_stream_preprocessing.py
│   ├── test_stream_stats.py
│   ├── test_stream_metrics.py
│   ├── test_stream_pipeline.py
│   ├── test_stream_trainer.py
│   ├── test_stream_visualise.py
│   ├── test_stream_edge_cases.py
│   └── ...                         # Original group test files
├── demo/
│   ├── stream_demo.ipynb           # Assignment 2.2 streaming demonstration
│   ├── sample_data.csv             # 200-row synthetic customer dataset used in the demo
│   ├── quickstart.ipynb            # Original group demonstration
│   └── sample_data.tsv
├── benchmark/
│   ├── run_stream_benchmarks.py    # Streaming loop/NumPy and buffer-size comparisons
│   ├── run_benchmarks.py           # Original group benchmark runner
│   └── results.md                  # Historical group benchmark results
├── pyproject.toml
├── BUGLOG.md
├── CONVENTIONS.md
└── README.md
```

---

## 3. Installation & Setup

The package declares **Python 3.9+** and **NumPy 1.24+**. The latest reported verification used Windows and Python 3.14.0.

### Step 1: Clone the Individual Branch

The `--branch` option selects the assignment code:

```powershell
git clone --branch assignment-2.2 https://github.com/Legacy164/NumCompute_2.git
cd NumCompute_2
git branch --show-current
```

The last command should print `assignment-2.2`. If you already cloned the fork, run `git switch assignment-2.2` from its root folder.

### Step 2: Install

From the repository root:

```powershell
python -m pip install -e ".[dev]" matplotlib ipykernel pytest-cov
```

In VS Code, select the same Python environment for the terminal and the notebook kernel.

### Step 3: Run the Tests

```powershell
python -m pytest tests -q
```

Expected output:

```text
339 passed
```

---

## 4. Quickstart Examples

Numeric features use shape `(n_samples, n_features)`. Labels use shape `(n_samples,)`. Run all examples from the repository root.

### 4.1 Streaming Pipeline with Logging (`numcompute_stream.stream`)

```python
from numcompute.io import iter_csv_chunks
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

pipe = Pipeline([
    ("impute", Imputer()),
    ("tree", DecisionTreeClassifier(max_depth=3, buffer_size=60)),
])
trainer = StreamTrainer(pipe)

for chunk in iter_csv_chunks(
    "demo/sample_data.csv",
    chunksize=30,
    skip_header=1,
    usecols=(0, 1, 4),
):
    X = chunk[:, :2]
    y = chunk[:, 2]
    print(trainer.fit_chunk(X, y))
```

`fit_chunk()` evaluates a chunk before training on it. Accuracy is `None` during warm-up, before a tree is available. Later log entries contain chunk accuracy, cumulative accuracy, row counts and array-memory measurements.

### 4.2 Buffered Decision Tree (`numcompute_stream.tree`)

```python
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier, entropy

rng = np.random.default_rng(0)
X = rng.normal(size=(1200, 3))
y = (X[:, 0] > 0).astype(int)

tree = DecisionTreeClassifier(max_depth=5, buffer_size=500, impurity=entropy)
for start in range(0, len(X), 200):
    tree.partial_fit(X[start:start + 200], y[start:start + 200])

print(tree.predict(X[:5]))
```

### 4.3 Incremental Preprocessing (`numcompute_stream.preprocessing`)

```python
import numpy as np
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler

scaler = StandardScaler().partial_fit([[1.0], [3.0]])
scaler.partial_fit([[5.0], [7.0]])
print("Running mean:", scaler.mean_, "| Variance:", scaler.var_)

minmax = MinMaxScaler().partial_fit([[2.0], [4.0]])
minmax.partial_fit([[0.0], [8.0]])
print("Scaled:", minmax.transform([[0.0], [4.0], [8.0]]).ravel())

imputer = Imputer(fill_value=-1).partial_fit([[2.0, np.nan]])
imputer.partial_fit([[4.0, 8.0], [6.0, 12.0]])
print("Filled:", imputer.transform([[np.nan, np.nan]]))

encoder = OneHotEncoder(handle_unknown="ignore").partial_fit(["blue", "red"])
encoder.partial_fit(["amber"])
print("Columns:", encoder.get_feature_names_out())
```

### 4.4 Streaming Statistics (`numcompute_stream.stats`)

```python
from numcompute.io import iter_csv_chunks
from numcompute_stream.stats import StreamStats

age_stats = StreamStats([0, 20, 40, 60, 80, 100])
for chunk in iter_csv_chunks("demo/sample_data.csv", 30, skip_header=1, usecols=(0,)):
    age_stats.update_stats(chunk)

print("Mean:", age_stats.mean, "| Variance:", age_stats.var())
print("Quartiles:", age_stats.quantile([0.25, 0.5, 0.75]).ravel())
print("Histogram:", age_stats.histogram())
```

### 4.5 Streaming Metrics & ROC AUC (`numcompute_stream.metrics`)

```python
from numcompute_stream.metrics import StreamingAUC, StreamingMetrics

cumulative = StreamingMetrics(labels=(0, 1))
rolling = StreamingMetrics(labels=(0, 1), window_size=60)
for metric in (cumulative, rolling):
    metric.update([0, 1, 1, 0], [0, 1, 0, 0])
print(cumulative.result()["accuracy"], cumulative.result()["confusion_matrix"])

auc = StreamingAUC().update([0, 0], [0.1, 0.4])
print("One class only:", auc.result())
auc.update([1, 1], [0.35, 0.8])
print("Both classes:", auc.result())
```

### 4.6 Visualisation (`numcompute_stream.visualise`)

```python
from numcompute_stream.visualise import (
    compare_models, plot_metric_over_time, plot_predictions_vs_ground_truth,
)

plot_metric_over_time([0.57, 0.70, 0.67, 0.70], "Accuracy per chunk", "Accuracy")
compare_models([0.57, 0.63, 0.64], [0.60, 0.66, 0.69], labels=("30-row buffer", "60-row buffer"))
plot_predictions_vs_ground_truth([0, 1, 1, 0], [0, 1, 0, 0], save_path="latest_chunk.png")
```

Each function returns a matplotlib figure and accepts optional `save_path` and `show` keyword arguments. Notebook figures can be displayed inline.

---

## 5. Streaming API Overview

| Module | Primary Classes & Functions | Behaviour & Algorithmic Notes |
| :--- | :--- | :--- |
| **`numcompute_stream.tree`** | `DecisionTreeClassifier`, `gini`, `entropy`, `best_numeric_split` | `fit()` builds a tree from scratch and clears the buffer. `partial_fit()` copies each chunk into the buffer and rebuilds the tree once at least `buffer_size` rows have arrived. Gini and entropy split scores use NumPy cumulative class counts. |
| **`numcompute_stream.preprocessing`** | `StandardScaler`, `MinMaxScaler`, `Imputer`, `OneHotEncoder` | `fit()` resets learned state; `partial_fit()` extends it. Streaming imputation supports mean and constant strategies. The encoder adds categories incrementally. |
| **`numcompute_stream.stats`** | `StreamStats` | `update_stats(X_chunk)`, `mean`, `var()`, `quantile(q)` and `histogram()`. Histogram edges stay fixed across chunks, and statistics omit NaNs. |
| **`numcompute_stream.metrics`** | `StreamingMetrics`, `StreamingAUC` | `update()`, `result()` and `reset()`. Accuracy, per-class precision, recall, F1 and confusion matrix; cumulative by default, rolling with `window_size`. |
| **`numcompute_stream.pipeline`** | `Pipeline` | Buffers raw chunks and updates preprocessing and the tree together at retraining boundaries. |
| **`numcompute_stream.stream`** | `StreamTrainer` | `fit_chunk()` and `score_chunk()`. Scores each chunk before training and logs per-chunk accuracy, cumulative accuracy and memory. |
| **`numcompute_stream.visualise`** | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth` | Reusable matplotlib figures with optional saving or inline display. |

### 5.1 Buffered Tree

- `fit(X, y)` builds a tree from scratch and clears pending buffered data.
- `partial_fit(X, y)` copies each chunk into the buffer. Once at least `buffer_size` rows have arrived, it rebuilds the tree on all buffered rows and clears the buffer.
- Between retrains, `predict(X)` uses the most recently trained tree. Prediction before the first training raises an error.
- `max_depth` accepts `None` or a nonnegative integer; zero produces a leaf. `min_samples_split` must be an integer of at least two.
- `buffer_size` and optional `max_features` must be positive integers. `max_features=None` uses all features; `random_state` controls feature subsampling.
- Pass `impurity=gini` or `impurity=entropy`, using the functions from `numcompute_stream.tree`, without quotation marks. Custom impurity callables are also supported.

Gini and entropy split scores use NumPy cumulative class counts. Prediction routes rows together at each node. Loops remain for tree structure, feature search and custom impurity functions. Equal computed split scores select the lowest candidate threshold. Majority-label ties follow NumPy's sorted label order.

### 5.2 Preprocessing & Pipeline

`StandardScaler`, `MinMaxScaler` and `Imputer` support `fit`, `partial_fit` and `transform`. `fit` resets learned state; `partial_fit` extends it. Streaming imputation supports mean and constant strategies.

`OneHotEncoder.partial_fit()` adds categories incrementally. Adding a sorted category can change the encoded column order and output width.

The streaming pipeline buffers raw chunks and updates preprocessing and the tree together at retraining boundaries. This keeps the active tree consistent with its feature transformation. Its final step is a buffered decision tree; preceding steps must support `partial_fit` and `transform`.

### 5.3 Statistics & Metrics

`StreamStats(bin_edges)` provides `update_stats(X_chunk)`, `mean`, `var()`, `quantile(q)` and `histogram()`. Histogram edges stay fixed across chunks, and statistics omit NaNs.

`StreamingMetrics` provides `update(y_true, y_pred)`, `result()` and `reset()`. Results include accuracy, per-class precision, recall, F1 and the confusion matrix. Confusion-matrix rows are actual classes and columns are predicted classes. Set `window_size` for rolling metrics; the default is cumulative.

`StreamingAUC` accepts true labels and caller-supplied scores through `update(y_true, y_score)`. It supports cumulative or rolling history and returns NaN until both binary classes are present. The current tree returns class labels, not probabilities, so the notebook's AUC example uses separate demonstration scores.

---

## 6. Performance Benchmarks

Run the streaming benchmarks as a Python module:

```powershell
python -m benchmark.run_stream_benchmarks
```

### Benchmark Method

- **Loop vs NumPy**: accumulates binary confusion matrices across 100 chunks of 1,000 rows. Both implementations must give equal results before the best of three timing runs is reported.
- **Buffer size**: uses the synthetic CSV, the same imputer and a tree with `max_depth=3`. Both configurations are evaluated on the same 140 rows, starting with the third chunk, and predictions precede training on each evaluated chunk.
- **Environment**: latest reported results on Windows with Python 3.14.0.

### Loop vs NumPy

| Implementation | Time |
| :--- | ---: |
| Python loop | 10.48 ms |
| NumPy updates | 5.97 ms |

NumPy updates were approximately **1.76 times faster** in this run.

### Buffer Size

| Buffer size | Accuracy | Evaluated rows | Retrains | Mean retraining update time |
| :--- | ---: | ---: | ---: | ---: |
| 30 | 67.1% | 140 | 6 | 0.88 ms |
| 60 | 69.3% | 140 | 3 | 1.36 ms |

Timing covers the pipeline retraining update, including preprocessing. The larger buffer retrains less often but takes longer per retraining update in this run.

*Note: Timings vary with the environment and run. Notebook cumulative curves can cover different evaluation rows because the two buffers have different warm-up periods; use the benchmark's shared evaluation rows for the accuracy comparison. The original group benchmarks in [`benchmark/results.md`](benchmark/results.md) describe the earlier batch package and are separate from these streaming results.*

---

## 7. Testing & Quality Assurance

Latest reported full-suite result: **339 tests passed**. This includes the inherited group tests and the streaming extension's tests.

```powershell
python -m pytest tests -q
python -m pytest tests --cov=numcompute_stream --cov-report=term-missing -q
```

### Streaming Package Coverage

Statement coverage for `numcompute_stream`, measured using the full suite:

| Module | Coverage |
| :--- | ---: |
| `metrics.py` | 91% |
| `pipeline.py` | 80% |
| `preprocessing.py` | 91% |
| `stats.py` | 89% |
| `stream.py` | 93% |
| `tree.py` | 87% |
| `visualise.py` | 100% |
| **Streaming package total** | **89%** |

This percentage applies to the streaming package, not the whole repository. It measures executed statements, not branch coverage or proof of correctness.

### Edge Cases Covered

- **Tree**: impurity calculations, split ties, constant features, depth limits, feature selection, invalid tree parameters and prediction order.
- **Buffering**: retraining at the buffer limit, copies of buffered input, feature-count mismatches and `fit()` resets.
- **Preprocessing**: incremental statistics, tiny-scale values, missing values, custom missing-value sentinels and category expansion.
- **Metrics**: cumulative and rolling metrics, unknown labels, AUC ties, NaN scores and reset.
- **Pipeline & trainer**: synchronised preprocessing and tree updates, and scoring before each update.
- **Visualisation**: figure creation and saving.

---

## 8. Interactive Demo Notebook

Open [demo/stream_demo.ipynb](demo/stream_demo.ipynb), restart the kernel and run all cells in order.

The notebook demonstrates:
1. CSV inspection and chunked loading with the custom `io.py` reader.
2. Incremental preprocessing with `partial_fit()`.
3. Streaming statistics with fixed histogram edges.
4. Buffered training of two pipelines (30-row and 60-row buffers) with per-chunk logs.
5. Cumulative and rolling classification metrics, and a ROC AUC example.
6. Plots of accuracy per chunk, buffer-size comparison and predictions versus ground truth.
7. Streaming benchmarks.

The CSV contains 200 synthetic customer records. The training example uses age and income as inputs and churn as the target. Thirty-row chunks produce six chunks of 30 and a final chunk of 20. Missing input values are handled by the imputer.

---

## 9. Assignment 1 Corrections

Corrections on the individual branch address distance precision, standardisation at tiny scales, scaler range validation, compatibility with minimal external transformers and unnecessary quadratic finite-difference storage. Regression tests were added with those changes. The original group repository is separate from these fork-branch changes.

---

## 10. Limitations & Recorded Demo

- Buffered retraining rebuilds the tree periodically; it does not update individual tree nodes online. Previous training rows are discarded when the buffer is cleared.
- The tree requires finite numeric features. Use preprocessing to handle missing or categorical inputs.
- Exact quantiles retain observations, and cumulative AUC retains labels and scores. Their memory usage grows with the stream; rolling AUC bounds its history.
- Logged memory values cover chunk arrays and pending raw buffers, not total model or process memory.
- The dataset is small and synthetic. Its accuracy does not establish performance on real customer data.
- The video shows an earlier version. Later validation, vectorisation and plotting-test changes leave the demonstrated benchmark accuracies unchanged. Test counts and timings in the video describe the recorded version; the results above describe the later reported run.

---

## 11. Authorship & Original Group Work

| Item | Details |
| :--- | :--- |
| **Individual student** | Tanmay Patel ([`@Legacy164`](https://github.com/Legacy164)) |
| **Individual work** | `numcompute_stream/` package, streaming tests, `demo/stream_demo.ipynb`, `benchmark/run_stream_benchmarks.py` and Assignment 1 corrections, all on `assignment-2.2` |
| **Original group package** | `numcompute/` by Divy Viradiya, Mihir Patel, Hetsi Patel and Tanmay Patel |

This branch builds on the group's work; it does not claim individual authorship of the inherited code. Original contribution details remain in the [group README](https://github.com/DivyViradiya/NumCompute/blob/main/README.md) and Git history.

---

## 12. License

Developed for educational purposes as part of the Adelaide University *Programming for AI* curriculum. Released under the MIT License.

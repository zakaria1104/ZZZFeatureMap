# Fields to Qubits: Quantum Machine Learning for Crop Classification in Morocco

> **Can a handful of qubits help distinguish alfalfa, maize, and wheat?**  
> A classical–quantum benchmark for three-class crop classification using satellite-derived features from the Doukkala region of Morocco.

This project explores how quantum machine learning can be applied to a practical agricultural classification task. It compares classical machine-learning baselines with quantum-kernel and variational quantum classifiers, using the official train, validation, and test partitions supplied for the **Qiskit Fall Fest × EMSI × Quantum Africa** crop-classification challenge.

The goal is not to assume that quantum models are better. It is to evaluate them under a controlled protocol, understand their class-level behavior, and identify where quantum methods may—and may not—be useful.

---

## Project at a glance

| Item | Description |
|---|---|
| Task | Three-class crop classification |
| Classes | Alfalfa, maize, wheat |
| Data | Satellite-derived numeric features (Sentinel-2 spectral bands and NDVI, following the organizer's feature manifest) |
| Region | Doukkala, Morocco |
| Training split | 914 labeled samples |
| Validation split | 150 labeled samples (50 per class) |
| Test split | 221 samples; labels are not available for local evaluation |
| Quantum input | 4 PCA components encoded into 4 qubits |
| Framework | Qiskit and Qiskit Machine Learning |
| Primary metric | Validation macro-F1 |

### Training-set class distribution

| Crop | Samples | Share |
|---|---:|---:|
| Wheat | 425 | 46.50% |
| Alfalfa | 247 | 27.02% |
| Maize | 242 | 26.48% |

The training set is moderately imbalanced, while the official validation set is balanced. The official partitions are preserved; this project does not create a new random train/validation/test split.

---

## Methodology

The experiment follows a leakage-aware workflow:

1. **Load the official splits.** The runner first attempts to use the organizer's `challenge.data.load_data()` loader. If it is unavailable, it looks for the supplied CSV split files.
2. **Preprocess using training data only.** A `StandardScaler` and a four-component PCA transformation are fitted on the training split. The fitted transformations are then applied unchanged to validation and test data. A second scaler standardizes the PCA representation.
3. **Encode the compact representation.** The four PCA features are mapped to bounded rotation angles using \(x_q = \pi\tanh(x)\). The same transformed four-dimensional inputs are used by the quantum models and the matched classical SVM control.
4. **Train classical and quantum models.** The benchmark includes the organizer's RBF-SVM reference (when available), a matched RBF-SVM, a four-qubit quantum-kernel SVM, and variational quantum classifiers.
5. **Select and evaluate using validation data.** Macro-F1 is the primary selection metric; accuracy is used as a secondary criterion. The test split is used only to generate prediction files.

### Models

| Model | Input | Description |
|---|---|---|
| Organizer reference RBF-SVM | 13 Original features | The challenge's classical reference implementation, when its module is available. A clearly named RBF-SVM fallback is used if it is not. |
| Matched classical RBF-SVM | 8 PCA features | Classical RBF kernel trained on the exact same compact, transformed inputs as the quantum models. A small grid over `C` and `gamma` is evaluated on validation data. |
| ZZ fidelity QSVM | 8 PCA features / 4 qubits | A ZZ feature map generates quantum states; an exact statevector fidelity kernel is computed and passed to a classical SVM classifier. Feature-map repetitions and SVM `C` are compared. |
| VQC / QNN with ZZ map | 8 PCA features / 4 qubits | A variational quantum classifier using a ZZ feature map, a `RealAmplitudes` ansatz, and SPSA optimization. |
| VQC / QNN with Z map | 8 PCA features / 4 qubits | The same variational workflow with a Z feature map, enabling a feature-map comparison. |

For the variational models, the computational-basis outcomes are mapped to the three crop classes using the explicit convention `outcome % 3`. This is an implementation choice and should be considered when interpreting or extending the architecture.

The SVM-based models use class weighting where supported by the implementation. The validation split is never resampled or rebalanced.

---

## Evaluation protocol

- **Primary metric:** macro-F1, which gives equal weight to each crop class.
- **Additional metrics:** accuracy, macro-precision, macro-recall, per-class precision/recall/F1, and confusion matrices.
- **Efficiency and circuit diagnostics:** fit time, prediction time, qubit count, circuit depth, two-qubit gate count, and trainable quantum parameter count where applicable.
- **Model selection:** validation macro-F1, with accuracy as a tie-breaker.
- **Test policy:** test labels are not accessed. The test split is inference-only, so no test accuracy or test macro-F1 is reported.

### Reported validation snapshot

The project presentation reports the following reference values:

| Model | Validation macro-F1 | Validation accuracy |
|---|---:|---:|
| Repository RBF-SVM baseline | 0.8797 | 88.00% |
| Starter QNN | 0.8557 | 85.56% |
| Project model (reported result) | approximately 0.86 | approximately 87% |

These are presentation-reported figures, not a substitute for the run-specific output of the benchmark script. For reproducible reporting, run the code and use `validation_leaderboard.csv` as the canonical source of results. The result should always be associated with its exact model configuration and seed.

**Interpretation:** the reported project result is competitive with the starter QNN but does not exceed the repository's RBF-SVM reference. This experiment does not establish quantum advantage.

---

## Repository contents

```text
.
├── README.md
├── SVM_and_QSVM.ipynb
├── QNN.ipynb
├── Fake_Backend_QSVM.ipynb
├── data/                         # optional location for organizer-provided CSV files
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── outputs/                      # generated experiment results
```

The dataset is not bundled with this project. Obtain it from the organizers' challenge repository and follow its data-use instructions. Do not commit data files unless redistribution is explicitly permitted.

Challenge repository: [Qiskit Fall Fest — EMSI × Quantum Africa](https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica)

---

## Installation

Use a clean Python environment. The Qiskit versions used by this project are pinned in `requirements-qnn-qsvm.txt` to avoid mixing incompatible Qiskit releases.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-qnn-qsvm.txt
```

For the notebook interface, install JupyterLab as well:

```bash
python -m pip install jupyterlab
jupyter lab
```

---

## Data layout and execution

The runner prefers the organizer's `challenge.data.load_data()` function. If that import is unavailable, it searches for these CSV files in the project root, `data/`, or `challenge/data/`:

- `train.csv` — must contain numeric features and the `target` column.
- `validation.csv` (also accepts `valid.csv` or `val.csv`) — must contain numeric features and the `target` column.
- `test.csv` — must contain the same numeric feature columns. Test labels are not required or used.

An optional `sample_id` column is preserved in prediction exports. When using the organizer's loader, the feature manifest determines the feature columns.

Run the full benchmark, including both VQC feature-map configurations:

```bash
python challenge_qnn_qsvm_benchmark.py \
  --output-dir outputs/qnn_qsvm_benchmark \
  --qnn-maxiter 100
```

Run only the classical baselines and QSVM:

```bash
python challenge_qnn_qsvm_benchmark.py \
  --output-dir outputs/qsvm_only \
  --skip-qnn
```

The QNN seed can be changed with `--qnn-seed` (default: `42`). To examine seed sensitivity, run separate experiments with different seeds and separate output directories. Do not describe a single-seed result as a multi-seed mean.

---

## outputs

The benchmark creates the selected output directory and writes:

| File | Contents |
|---|---|
| `validation_leaderboard.csv` | Validation metrics and model/runtime metadata for each evaluated model |
| `classical_svm_grid.csv` | Classical RBF-SVM configurations tested |
| `qsvm_grid.csv` | QSVM feature-map repetitions and SVM configurations tested |
| `qnn_objective_history.csv` | VQC optimization objective history, when QNN training is enabled |
| `validation_diagnostics.json` | Confusion matrices and per-class classification reports |
| `validation_predictions_*.csv` | Validation predictions for inspection and auditing |
| `test_predictions_*.csv` | Test predictions from individual evaluated models |
| `final_quantum_test_predictions.csv` | Test predictions from the best quantum-containing model selected using validation macro-F1 |
| `run_summary.json` | Split information, preprocessing protocol, selected quantum model, and evaluation policy |

The test prediction files are submissions/inference outputs only. Since the test labels are unavailable, they do not provide a test-set performance score.

---

## Quantum execution and limitations

The core benchmark uses **exact, noiseless local statevector simulation** for the quantum kernel and variational circuits. It is not an execution on a physical IBM Quantum Processing Unit (QPU), and it does not demonstrate quantum advantage.

The project presentation also reports a separate experiment using the **IBM Marrakesh FakeBackend**, with approximately 0.85 macro-F1 and 85% accuracy. This is a noisy-backend simulation, not a physical-hardware result. The current core benchmark script does not reproduce that experiment; include the corresponding backend code, configuration, and logs in the repository before treating the figure as independently reproducible.

Other limitations to keep in mind:

- Four PCA components provide a compact representation, but may discard information useful for crop discrimination.
- The dataset represents the released Doukkala cohort and its supplied imagery period; results should not automatically be generalized to every Moroccan region, season, or crop condition.
- The validation set is relatively small. Per-class metrics and seed sensitivity should be considered alongside the headline score.
- The class-readout rule `outcome % 3` is a design choice, not a uniquely prescribed quantum measurement scheme.
- A classical model currently provides the strongest reported validation reference. Quantum models should therefore be assessed on predictive quality, resource cost, and robustness—not on the assumption that quantum processing is inherently superior.

---

## Reproducibility checklist

Before publishing or comparing a final result:

- [ ] Record the exact commit, package versions, model configuration, and QNN seed.
- [ ] Confirm that the organizer-provided train/validation/test partitions were used unchanged.
- [ ] Confirm that scaling and PCA were fitted on training data only.
- [ ] Report validation macro-F1, accuracy, and per-class metrics.
- [ ] Include confusion matrices and runtime/circuit-resource information.
- [ ] Repeat stochastic QNN runs with multiple seeds before reporting mean and standard deviation.
- [ ] Keep test labels unused and do not report test metrics without ground truth.
- [ ] Distinguish local statevector simulation, FakeBackend simulation, and real QPU execution.

---

## Acknowledgements

Developed for the **Qiskit Fall Fest × EMSI × Quantum Africa** crop-classification challenge. The challenge dataset and fixed evaluation splits are provided by the organizers. This project uses Qiskit and Qiskit Machine Learning to investigate quantum-kernel and variational approaches alongside classical baselines.

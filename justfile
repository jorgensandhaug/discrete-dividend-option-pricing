set shell := ["zsh", "-cu"]

sync:
    uv sync --all-groups --extra torch --extra mlx --locked

fmt:
    python3 tools/dev.py -- ruff format .

engine-revision:
    test "$(tr -d '[:space:]' < config/engine-revision.txt)" = "$(git -C ../finoptions-engine rev-parse HEAD)"

check: engine-revision
    python3 tools/dev.py -- ruff format --check .
    python3 tools/dev.py -- ruff check .
    python3 tools/dev.py -- ty check --error-on-warning src tests tools benchmarks

test:
    python3 tools/dev.py -- pytest -q

full: check test
    uv build

bench-neural variant="nn_bin" rows="100000" validation_rows="10000" epochs="10":
    python3 tools/dev.py -- python benchmarks/neural_training.py --variant {{variant}} --rows {{rows}} --validation-rows {{validation_rows}} --epochs {{epochs}}

bench-inference variant="nn_bin" rows="100000" batch_size="8192":
    python3 tools/dev.py -- python -m benchmarks.neural_inference --variant {{variant}} --rows {{rows}} --batch-size {{batch_size}}

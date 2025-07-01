# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 開発原則
ユーザー＝意思決定 / Claude＝実装 / Gemini＝助言（Google 検索）

## 実践ガイド
- ユーザーの要求を受けたら **即座に Gemini と壁打ち**
  - `gemini -p "<質問内容>"`
- **Claude Code 内蔵 WebSearch ツールは使わない**
- Gemini がエラー時は聞き方を変えてリトライ
- Geminiは長いコンテキストを読むのが得意なので、大量のログ解析や長文ドキュメントの要約などを任せる

## Common Development Commands

### Environment Setup
```bash
# Initial setup (installs system dependencies, creates venv, installs Python packages)
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or use: source env.sh
```

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run specific test categories
./run_tests.sh unit          # Fast unit tests only
./run_tests.sh integration   # Integration tests
./run_tests.sh performance   # Performance tests
./run_tests.sh all          # All tests

# Run a single test file
pytest tests/test_collision.py -v

# Run tests with specific markers
pytest -m "not slow and not camera" -v
```

### Type Checking
```bash
# Run mypy type checker
mypy src/

# Check specific module with strict settings
mypy src/config.py --strict
```

### Running the Application
```bash
# Run the main collision detection demo
python demo_collision_detection.py

# Run with specific options (check demo files for available arguments)
python demo_collision_detection.py --debug
```

## Architecture Overview

### Pipeline Architecture
The system implements a real-time gesture-to-sound pipeline with these phases:

1. **Input Phase** (`src/input/`): Captures RGB-D data from Orbbec camera, converts to point clouds
2. **Detection Phase** (`src/detection/`): Uses MediaPipe to detect hand landmarks in 2D, projects to 3D using depth
3. **Mesh Phase** (`src/mesh/`): Generates terrain mesh using Delaunay triangulation
4. **Collision Phase** (`src/collision/`): Detects collisions between hand points and terrain mesh
5. **Sound Phase** (`src/sound/`): Synthesizes audio based on collision parameters using Pyo

### Key Design Patterns

- **Facade Pattern**: Each phase has a facade class (e.g., `InputFacade`, `DetectionFacade`) providing a simplified interface
- **Strategy Pattern**: Multiple implementations for components (e.g., different audio backends)
- **Type Safety**: Shared types in `src/types.py`, strict mypy checking for core modules
- **Performance Optimization**: Numba JIT compilation for critical paths, 40ms latency target

### Module Dependencies
```
input → detection → mesh → collision → sound
         ↓           ↓        ↓          ↓
      types.py (shared type definitions)
         ↓
    constants.py (configuration constants)
```

### Testing Strategy
- Unit tests for individual components
- Integration tests for phase interactions
- Performance tests with latency measurements
- Mock implementations for hardware dependencies (camera, audio)

### Configuration Management
- Central configuration in `src/config.py` using dataclasses
- Environment-based settings with defaults
- Resource management through `src/resource_manager.py`

### Performance Considerations
- Target: 40ms end-to-end latency
- Numba JIT compilation enabled for collision detection
- Optional GPU acceleration with CuPy
- Memory optimization for point cloud processing
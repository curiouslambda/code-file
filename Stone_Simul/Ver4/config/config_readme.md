# Config

### 목적
- 실험 파라미터와 출력 경로를 YAML로 관리

### 주요 파일
- `dp_config.yaml`
  - `gamma`, `theta`, `max_iterations`
  - `log_file`, `value_curve_plot`, `value_heatmap_plot`, `value_save_path`
- `mcts_config.yaml`
  - `c_puct`, `max_simulations`, `n_episodes`
  - `log_file`

### 사용
- `load_config('config/dp_config.yaml')`
- `load_config('config/mcts_config.yaml')`


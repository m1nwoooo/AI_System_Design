# AI_System_Design
Iterative design space exploration for optimal neural network layer partitioning and placement on multi-core NoC architecture to maximize throughput and power efficiency.

구체적인 개발과정과 Waveform 등은 **AI_System_Design.pdf**로 첨부하였습니다.


# NoC-based AI Accelerator Design Space Exploration 🚀

Network-on-Chip 기반 AI 가속기를 위한 반복적 설계 공간 탐색 시스템 구현

## 📋 프로젝트 개요

36개 core와 12개 DRAM으로 구성된 8×6 NoC HW에서 BERT 모델의 Layer Group들을 Partitioning & Routing하여 Throughput과 Power Efficiency를 극대화하는 DSE System 구축

## 🛠️ 개발 과정 및 핵심 모듈

### 1️⃣ HW Topology Parsing (Adaptive)
- **hardware.json 파싱**: component_mapping, nodes, routers
- **HW 구성 자동 감지**: 48-router(8×6), 64-router(8×8) 등

### 2️⃣ Iteration 1: 균등 초기 배치
- **Min Core**: 각 태스크당 MIN_CORES_PER_TASK(3개) 할당
- **Vertical Strategy**: DRAM 근접성 우선 배치
- 모든 36개 코어 균등 분배
- 초기 베이스라인 성능 측정

### 3️⃣ Iteration 2: Weight 기반 재분배
- **Load + Compute Time 기반 Weight**: `allocation.csv`, `execution_time.csv` 기반 weight 분배
- **중복 감지 및 Boost**: 이전 iteration과 동일 시 병목 task에 10% 추가 할당
- Compute/Memory bound 판별 후 전략 선택 (Horizontal/Vertical)

### 4️⃣ Iteration 3+: link load & Bottleneck 분석
- **Link Bandwidth 분석**: `link_load.csv`에서 태스크별 평균 대역폭 효율 계산
- **최악 대역폭 태스크 Boost**: 가장 낮은 대역폭 태스크에 우선 코어 할당
- **실행 시간 병목 분석**: 최대 실행 시간 태스크의 Compute/Comm bound 판별
- Placement 전략 동적 조정: COMM bound → Vertical, COMPUTE bound → Horizontal

### 5️⃣ Regression 검사 및 수렴 감지
- **성능 저하 감지**: 이전 iteration 대비 시간 5% 이상 증가 시
- **자동 roll-back**: 이전 코어 할당으로 복구
- **수렴 판단**: 개선율 2% 미만 시 최적화 종료
- Sampling efficiency 향상
─────────────────────────────────┘


## 📈 최적화 전략

| Iteration | Partitioning 전략 | Placement 전략 | 핵심 메트릭 |
|-----------|------------------|----------------|------------|
| 1 | Uniform Split | Vertical (DRAM 근접) | 베이스라인 |
| 2 | Load+Compute Time 가중치 | Compute/Mem bound 판별 | MAC 균형 |
| 3+ | Link BW 분석 + Boost | Bottleneck 기반 동적 | 병목 제거 |

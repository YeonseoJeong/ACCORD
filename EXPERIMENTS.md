# ACCORD 실험 가이드

논문 *Learning to Coexist*의 실험(시나리오, 비학습 기준선, 학습 방법, 결과표)을 처음부터 재현하는 절차입니다.
명령어는 모두 이 폴더(`ACCORD/`)에서 실행합니다. 파일별 설명과 개발 과정의 기록은 `README.md`에 있습니다.

## 0. 환경 준비 (한 번만)

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy matplotlib
```

CPU만으로 충분합니다. 아래 소요 시간은 16코어 Apple M4 Max 기준입니다. 코어가 적으면 `--workers`(probe)와 `JOBS`(학습)를 줄이세요.

설치 확인용 짧은 실행입니다. 1분 안에 끝나야 합니다.

```bash
.venv/bin/python trainer.py --method ippo --iters 2 --envs 4 --out runs/_smoke
.venv/bin/python evaluate.py runs/_smoke --episodes 5
rm -rf runs/_smoke*
```

## 1. 실험 설정 (이미 기본값으로 들어 있음)

| 항목 | 값 | 위치 |
|---|---|---|
| 시나리오 | preset `hotspot`: RSRP_min −120 dBm, 비인접 외곽 셀 2개에 UE 70%의 고정 hotspot(반경 90 m), hotspot 피크 T/2 간격, d̄ = 2.2 Mbps, 커버리지 제약 (10f) 집행 | `ran_env.py`의 `PRESETS` |
| MLB 행동 | 셀마다 CIO를 −δ / 0 / +δ만큼 변경(증분) | `trainer.py`의 `HP.mlb_action` |
| 학습 설정 | λ는 PID Lagrangian(K_P = 20, K_I = 0.1, 초기 λ = 1), ES 엔트로피 0.01 → 0 선형 감쇠, MLB 엔트로피 고정, 엔트로피는 셀 평균, 1000 iteration | `trainer.py`의 `HP` |
| 평가 | 모든 방법이 같은 30개 episode(chunk k = 0…5, `seed = 1000·k`, chunk당 5개), 확률적 실행(행동을 샘플링, 논문 Algorithm 2) | `evaluate.py --stochastic` |
| 학습 seed | 환경 seed를 10,000,000부터 사용하므로 평가 episode와 겹치지 않음 | `trainer.py` |

**평가 episode와 seed 규칙은 바꾸지 마세요.** episode 간 편차가 커서, 방법마다 episode가 다르면 비교가 의미 없어집니다.

## 2. 비학습 기준선과 참고값 (약 15분)

All-on, rule-based, oracle 참고값을 30개 episode로 측정합니다.

```bash
.venv/bin/python probe.py --preset hotspot --seeds 30 --name probe_hotspot_d2.2
.venv/bin/python reference_table.py --probe results/probe_hotspot_d2.2.json
```

- 출력: `results/probe_hotspot_d2.2.json`, `results/reference_d2.2.md`, `results/reference_d2.2.tex`
- 결과표의 All-on, All-on + MLB rule, Rule-based, All-on + CIO oracle, Oracle reference 행이 여기서 나옵니다.
- Oracle 행은 미래 정보를 쓰는 휴리스틱이라 엄밀한 상한이 아닙니다.

## 3. 학습 방법 (Main table과 Ablation 표)

Main table:

```bash
PREFIX=main ITERS=1000 JOBS=10 ./run_batch.sh "ippo es_only_i es_cio0_i central" "1 2 3 4 5"
```

Ablation (같은 설정):

```bash
PREFIX=main ITERS=1000 JOBS=10 ./run_batch.sh "accord dtde_team dtde_sym dtde_wh0" "1 2 3 4 5"
```

- 각각 방법 4개 × 학습 seed 5개 = 20개 실행을 10개씩 병렬로 돌립니다.
- 실행당 1000 iteration입니다. PID multiplier로 약 400 iteration에 제약 경계(Ĉ ≈ V_max)에 도달하고, 남은 구간에서 ES 엔트로피 감쇠가 끝납니다.
- 10개 동시 실행이면 실행당 약 3.5시간, 20개 전체는 약 7시간입니다.
- 결과 위치: `runs/main_<method>_s<seed>/`

| 파일 | 내용 |
|---|---|
| `log.csv` | iteration별 V, Ĉ, λ, 켜진 셀 수, handover, θ = 0 비율 등 |
| `final.pt`, `ckpt_*.pt` | 학습된 네트워크 |
| `config.json` | 이 실행의 모든 설정 |
| `eval_stochastic_final.json` | 30개 episode 평가(절감, V̄, 만족 episode, CIO 분포 포함). 결과표에 쓰는 값 |
| `eval_stochastic_mlboff_final.json` | 같은 ES 정책에서 MLB만 끈 평가(학습된 MLB가 있는 방법만) |
| `eval_greedy_final.json` | 결정적(mode) 실행 평가. 확률적 실행과의 차이를 보는 참고용 |

이미 `final.pt`가 있는 실행은 학습을 건너뛰고 평가만 다시 합니다. 중간에 멈춰도 다시 실행하면 이어집니다.

방법 이름(코드의 method 이름 → 논문 표기)은 다음과 같습니다.

| 구분 | method | 논문 표기 | 내용 |
|---|---|---|---|
| 제안 | `ippo` | ACCORD | DTDE: 보상 분리, λ는 ES에만, actor와 critic 모두 자기 관측만 봄(ES는 θ 없음, MLB는 s·τ 없음) |
| Main | `es_only_i` | ES-only learning | 학습 ES + 데드존 rule MLB, 적분형 λ |
| Main | `es_cio0_i` | Learned ES, MLB off | 학습 ES + MLB 끔(CIO = 0), 적분형 λ. "MLB를 끄면 되지 않느냐"에 대한 비교 |
| Main (참고) | `central` | Centralized | 공동 보상, 하나의 agent가 두 변수를 결정(π(s\|x)·π(θ\|x, s)). O-RAN에서 배치할 수 없는 참고값 |
| Ablation | `accord` | ACCORD (CTDE) | 두 agent가 전체 상태를 보고, critic도 전체 상태를 봄 |
| Ablation | `dtde_team` | ACCORD (team reward) | 공동 보상 −(Ē + w_h H), λ는 두 agent 모두 |
| Ablation | `dtde_sym` | ACCORD (symmetric λ) | 보상 분리, λ는 두 agent 모두 |
| Ablation | `dtde_wh0` | ACCORD, w_h = 0 | MLB 보상에서 handover 비용 제거 |

- Ablation 행은 다른 접근법이 아니라 ACCORD의 설계 요소 하나를 바꾼 변형입니다.
- λ를 MLB에도 거는 변형(`dtde_team`, `dtde_sym`)에서는 MLB가 자기 관측으로 된 비용 critic을 따로 가집니다(DTDE 유지).
- 실행 폴더 이름은 코드의 method 이름을 그대로 씁니다(예: `runs/main_ippo_s1`이 ACCORD).
- `team`, `accord_sym`, `accord_no_theta_es`, `accord_no_tau_mlb`, `accord_wh0`는 이전의 CTDE 기반 변형입니다.
- multiplier는 PID Lagrangian이 기본입니다. 적분형 업데이트(논문 식 (38))는 `--lam_ctrl i`로 켭니다.
- 학습된 MLB가 없는 두 baseline(`es_only_i`, `es_cio0_i`)은 적분형 λ를 씁니다. ES 혼자 제약을 지켜야 해서 균형 λ가 약 8로 높습니다. ACCORD에 맞춘 PID 이득값(균형 λ 약 2)으로는 1000 iteration 안에 거기까지 오르지 못하기 때문입니다. PID로 돌린 `es_only`, `es_cio0`는 진단용입니다.

## 4. 결과표 만들기

```bash
.venv/bin/python collect_results.py --prefix main --probe results/probe_hotspot_d2.2.json
```

Main table(비학습 행, 한쪽만 학습하는 방법, ACCORD, Centralized, oracle 참고값)과 Ablation 표(ACCORD와 그 변형)가 따로 나오고, 각 표의 LaTeX 행도 함께 출력됩니다. 값은 seed 평균 ± 표준편차입니다.

학습 곡선(λ, Ĉ, 켜진 셀 수, θ = 0 비율)도 그릴 수 있습니다.

```bash
.venv/bin/python plot_training.py runs/main_ippo_s1 runs/main_es_only_s1 runs/main_es_cio0_s1 --out results/train_main.png
```

## 5. 추가 실험

**적분형 λ 업데이트 (ablation 행 ACCORD (integral λ))**

기본 multiplier는 PID입니다. 논문 원래의 적분형 업데이트(식 (38), η_λ = 0.5)는 초반에 λ가 5–6까지 쌓여서, 1000 iteration이 되어서야 제약 경계에 도달합니다. `run_batch.sh`는 학습 하이퍼파라미터를 넘기지 않으므로 직접 실행합니다.

```bash
for s in 1 2 3 4 5; do
  .venv/bin/python trainer.py --method ippo --iters 1000 --seed $s --save_every 100 \
      --lam_ctrl i --out runs/integral_ippo_s$s
  .venv/bin/python evaluate.py runs/integral_ippo_s$s --stochastic --threads 1
done
.venv/bin/python collect_results.py --prefix main --extra "ACCORD (integral λ)=integral:ippo"
```

**Sweep (논문 Sec. V-D)**

학습 실행과 같은 설정으로 비학습 참고값도 다시 재야 합니다.

| sweep | 학습 | 참고값 |
|---|---|---|
| ES epoch K | `PREFIX=k30 BASE="K=30" ITERS=1000 ./run_batch.sh "ippo es_only es_cio0" "1 2 3"` | `.venv/bin/python probe.py --preset hotspot --base K=30 --name probe_k30` |
| V_max | `PREFIX=vmax2 BASE="v_max=0.02" ITERS=1000 ./run_batch.sh "ippo es_cio0" "1 2 3"` | `.venv/bin/python probe.py --preset hotspot --base v_max=0.02 --name probe_vmax2` |
| 부하 d̄ | `PREFIX=d25 PRESET=hotspot_tight ITERS=1000 ./run_batch.sh "ippo es_cio0" "1 2 3"` | `.venv/bin/python probe.py --preset hotspot_tight --name probe_d25` |

- 결과표는 `collect_results.py --prefix <PREFIX> --probe <참고값 파일>`로 만듭니다.
- 주의: d̄ = 2.5와 V_max = 0.02는 seed 1 시험에서 500 iteration 안에 제약을 맞추지 못했습니다(λ가 계속 상승). 1000 iteration에서도 수렴하는지 `log.csv`의 λ와 Ĉ로 확인하세요.

## 6. 진단 도구

| 목적 | 명령 |
|---|---|
| MLB를 끄면 V가 얼마나 오르는가 | `.venv/bin/python evaluate.py runs/<run> --mlb_off` |
| 결정적(mode) 실행으로 평가 | `.venv/bin/python evaluate.py runs/<run>` (`--stochastic`을 붙이면 결과표 기준인 확률적 실행) |
| 커버리지 위반 확인 | `.venv/bin/python coverage_check.py runs/<run> ...` |
| CIO 한 단계가 보상을 바꾸는가 | `.venv/bin/python gradient_check.py` |
| 한 episode의 시간 궤적 | `.venv/bin/python trace_plot.py --preset hotspot` |
| 비학습 방법으로 다른 설정 탐색 | `.venv/bin/python probe.py --preset hotspot --sweep demand_mbps=2.0,2.2,2.5` |

## 7. 알아 둘 점

- **결과를 읽을 때 꼭 함께 볼 지표**
  - V̄의 평균만으로는 부족하고, 30개 중 몇 episode가 V_max를 만족하는지도 보세요.
  - 학습된 MLB가 있는 방법은 MLB를 끈 평가와 비교해야 MLB가 실제로 일하는지 알 수 있습니다.
- **학습 중 V(`log.csv`)는 평가보다 높습니다.** 학습 중에는 확률적으로 행동하기 때문입니다. 판단은 평가 결과로 하세요.
- **논문 원래 설정과 다른 점**: MLB 행동(증분), η_λ와 초기 λ, ES 엔트로피 감쇠는 논문 Table II와 다릅니다. 원래 설정은 `--mlb_action abs`, `--eta_lam 0.01 --lam0 0`, `--ent_es_end` 생략으로 재현할 수 있지만, 그 경우 MLB가 움직이지 않고 학습이 붕괴할 수 있습니다(`README.md` 참고).
- **`README.md`의 Status 절은 개발 과정 기록입니다.** 그중 일부 실행은 이전 코드로 돌린 것이니, 논문 수치는 이 가이드의 절차로 새로 만든 결과를 쓰세요.

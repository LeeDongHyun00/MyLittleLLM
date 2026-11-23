# SeedLM-34: 교사가 뼈대마다 쓴 과제 — 코더는 과제를 외울 뿐, 구조를 고르는 법은 배우지 않는다

작성일: 2026-11-23. [보강 설계](seedlm_design_next.md) 11.3의 c단계다. [SeedLM-30~33](seed_lm30.md)에서 뼈대 탐색은 맞는 프로그램에 닿았지만, 맞는 새 구조를 어떤 코더도 그럴듯하게 보지 않아 고르지 못했다. 그래서 교사(Opus 5.5)가 뼈대마다 자연스러운 과제를 써서 코더에게 "이런 설명에는 이런 구조"를 가르쳐 봤다. 측정 전에 등록했다(`reports/seedlm/seedlm34_protocol.json`).

## 요약

- 교사 셋이 학습 프로그램에서 두 번 이상 나온 뼈대 204개에 과제를 4개씩, 모두 814개 썼다. 오염 검사에서 26개가 빠졌고, 기본 코더의 어휘에 없는 기호가 필요한 16개를 빼 772개(뼈대 203개)를 썼다. 넷째 교사(뼈대 68개)는 지침을 읽기 시작하자마자 API의 안전 필터에 걸려 멈췄고, 사용자의 결정으로 세 몫만으로 진행했다(등록 문서의 note_1).
- 같은 체크포인트(s12-a1~a3)에서 400단계씩, 뼈대 과제를 섞은 쪽(s34-k)과 섞지 않은 쪽(s34-o)을 쌍으로 미세조정했다.
- **주 측정**(개발 과제, SeedLM-32 시스템): k 65, o 65로 **지지되지 않았다**. 둘 다 지금 시스템(69)에 못 미쳐 채택할 것이 없다.
- **진단:** k 코더는 학습한 뼈대 과제 120개 중 90~95개를 쓴다(o는 16~21개). 과제는 배웠다. 그러나 처음 보는 과제에서는 혼자 푸는 수도 후보 순위도 바뀌지 않았다. 뼈대 후보의 확률은 크게 올랐는데(로그 확률 +8~12), 틀린 후보가 맞는 후보보다 더 올랐다. 코더는 보여 준 과제를 한 쌍씩 외우고, "이 구조들이 있다"는 것만 배웠다. 어떤 설명이 어떤 구조를 부르는지는 배우지 않았다.

## 1. 데이터

| 몫 | 뼈대 | 쓴 과제 | 비고 |
|---|---|---|---|
| 1 | 68 | 0 | 교사가 안전 필터에 걸려 시작 직후 멈춤 |
| 2 | 68 | 270 | 뼈대 k225는 자연스러운 과제가 둘뿐이라 둘 |
| 3 | 68 | 272 | |
| 4 | 68 | 272 | |
| 합계 | 272 | **814** | 오염 검사에서 26개 빠짐(여섯 단어 연속 24, 내용어 자카드 2) → 788개, 뼈대 203개 |

지침은 [교사 지침](seedlm34_teacher_guide.md)에 있다. 교사는 자기 몫의 뼈대와 뼈대마다 학습 과제 두 개만 봤다. 과제는 SeedLM-3의 교사 과제 검사를 통과해야 하고, 풀이의 뼈대가 배정된 뼈대와 정확히 같아야 하며, 학습 과제의 식이나 설명을 되풀이하면 안 된다(`scripts/seedlm34_task_check.py`). 세 몫 모두 마지막 검사에서 문제 0개였다. 교사들이 스스로 밝힌 약점은 구조가 좁은 뼈대들이다. 이항 분포꼴 곱, 구면 코사인 법칙, 구간 최솟값을 뺀 최댓값 같은 뼈대는 네 과제가 주제는 달라도 수학은 같다.

교사가 쓴 과제의 예:

| 뼈대 | 설명 | 풀이 |
|---|---|---|
| k228 | Two teams of equal size are matched by pairing the weakest player of the first team with the strongest of the second, ...; count the pairs won by the first team's player. | `sum(x > y for x, y in zip(sorted(a0), sorted(a1)[::-1]))` |
| k232 | Convert an hour of the day from 0 to 23 into a 12-hour clock pair (hour, 'AM' or 'PM'), ... | `(a0 % 12 if a0 % 12 else 12, 'AM' if a0 < 12 else 'PM')` |
| k200 | Given two equally long strings of digits, return the string made of the larger of the two digits at each position. | `''.join(max(a, b) for a, b in zip(a0, a1))` |

## 2. 두 갈래

`scripts/seedlm26_finetune.py`의 쌍 미세조정을 그대로 썼다(`--words sketch` / `--words none`).

| | s34-k1~k3 | s34-o1~o3 |
|---|---|---|
| 시작점 | s12-a1~a3 | 같음 |
| 데이터 | 기본 혼합 + 뼈대 과제 772개 × 보기 3개(2,316행)를 혼합의 10%로 | 기본 혼합만(기본 실행의 비율 그대로) |
| 단계, 학습률 | 400단계, 기본의 0.3배 | 같음 |
| 마지막 손실 | 0.150, 0.155, 0.162 | 0.090, 0.095, 0.098 |

400단계 동안 뼈대 과제 한 행은 평균 1.1번, 과제 하나는 보기 셋을 합쳐 3.3번 나온다.

## 3. 결과

| | k | o |
|---|---|---|
| **개발 과제, SeedLM-32 시스템 (주 측정)** | **65** (엔진 대비 고침 1, 깨뜨림 1) | **65** (고침 1, 깨뜨림 1) |
| 개발 과제, 후보만 | 58 | 59 |
| 교사 보류 139, 시스템 / 후보만 | 58 / 64 | 58 / 64 |
| 코더 혼자, 교사 보류 (시드 셋) | 57, 56, 64 | 58, 58, 58 |
| 코더 혼자, 개발 과제 | 59, 50, 55 | 56, 51, 58 |

시스템은 "엔진이 확신하면 엔진, 아니면 여섯 코더(s5-var, 미세조정한 셋, s18-p1, s18-p2)의 평균 로그 확률이 가장 높은 후보"다. 후보는 SeedLM-32에서 모은 것 그대로다(`reports/seedlm/selection34_{development,teacher}_{k,o}.json`).

## 4. 왜 바뀌지 않았나 (탐색, 개수만)

`scripts/seedlm34_diagnose.py`, `reports/seedlm/seedlm34_diagnosis.json`.

| 학습한 뼈대 과제 120개(무작위)를 혼자 푼다 | s34-k1 / k2 / k3 | s34-o1 / o2 / o3 |
|---|---|---|
| 폭 32의 첫 검증 프로그램이 세 입력 모두 맞음 | **90 / 93 / 95** | 19 / 16 / 21 |

| 맞는 후보가 있는 과제에서 가장 높은 맞는 후보의 순위 | 교사 보류 88 | 개발 67 |
|---|---|---|
| 1위인 과제, k / o | 64 / 64 | 58 / 59 |
| k에서 오름 / 내림 / 같음 | 8 / 7 / 73 | 4 / 2 / 61 |

| 미세조정한 세 코더의 로그 확률 변화 (k − o, 평균) | 교사 보류 | 개발 |
|---|---|---|
| 뼈대 후보, 맞음, 가르친 뼈대 | +9.0 | +9.2 |
| 뼈대 후보, **틀림**, 가르친 뼈대 | **+11.7** | **+11.7** |
| 뼈대 후보, 맞음 / 틀림, 안 가르친 뼈대 | +7.7 / +10.9 | +4.1 / +10.0 |
| 코더 빔 후보, 맞음 / 틀림, 가르친 뼈대 | +2.1 / +4.6 | +4.8 / +5.4 |
| 코더 빔 후보, 맞음 / 틀림, 안 가르친 뼈대 | +0.9 / +2.5 | +0.3 / +0.8 |

교사 보류 과제 중 뼈대 탐색으로만 맞는 후보가 생긴 21개 가운데 16개, 개발 과제 10개 가운데 8개는 맞는 후보의 뼈대가 가르친 203개 안에 있었다. 가르칠 구조는 제대로 골랐다는 뜻이다.

읽기는 이렇다.

1. **배우기는 했다.** 과제 하나를 보기 셋으로 세 번쯤 보여 주자 코더는 그 과제를 77% 쓴다. 이전에는 16%였다.
2. **배운 것은 과제 한 쌍씩이다.** 처음 보는 과제에서는 혼자 푸는 수가 평균 +1(교사 보류), −0.3(개발)이었다. 설명을 외우는 법칙(SeedLM-25, 26)이 구조에도 그대로 선다. 코더는 "설명 → 구조"를 일반 규칙으로 배우지 않고, 보여 준 설명과 프로그램의 짝을 외운다.
3. **구조의 확률은 맞는 것과 틀린 것에 함께 오른다.** 같은 뼈대의 맞는 후보와 틀린 후보는 고정 토큰을 공유하고 구멍만 다르다. 고정 토큰이 그럴듯해지면 둘 다 함께 오른다. 틀린 후보는 수가 훨씬 많고 처음에 더 그럴듯하지 않았으니 더 많이 올랐다. 그래서 순위는 거의 그대로이고(88개 중 73개, 67개 중 61개가 같은 순위), 여섯 코더 평균에 섞인 미세조정 셋의 변화는 맞는 후보를 앞으로 끌어내지 못했다.

## 5. 다음

같은 뼈대의 후보들은 구멍에서만 갈린다. 그 차이를 가려내는 질문은 "이 프로그램이 설명의 단어를 다 설명하는가"이고, 이는 코더의 방향(설명 → 프로그램)이 아니라 거꾸로 된 방향, P(설명 | 프로그램)이다. SeedLM-3의 검토기(coder-reviewer 재정렬, Zhang 외 2022)는 연산 프로그램 위에서 배워 개발 과제 67(엔진 65)이었고 설명을 외웠다. 지금은 pyx 프로그램, 교사 과제 약 3,000개, 바꿔 쓴 문장 2,400개, 뼈대 과제 772개가 있다. 다음 단계(SeedLM-35)는 pyx 검토기를 학습해, 코더와 다른 방향에서 같은 후보를 다시 매기는 것이다. 바꿔 쓴 문장은 검토기가 설명을 외우는 것을 막는 데 쓴다.

## 재현

```bash
python scripts/seedlm34_task_check.py --all --summary
for k in 2 3 4; do python scripts/seedlm34_task_check.py --part $k --summary --accept; done
python scripts/seedlm3_decontaminate.py --prefix sketch_tasks
for k in 1 2 3; do
  python scripts/seedlm26_finetune.py --base s12-a$k --name s34-o$k --words none --steps 400
  python scripts/seedlm26_finetune.py --base s12-a$k --name s34-k$k --words sketch --steps 400
done
python scripts/seedlm15_ensemble_scaling.py cache --coders s34-o1 s34-o2 s34-o3 s34-k1 s34-k2 s34-k3
for set in teacher development; do
  python scripts/seedlm32_score.py --set $set --scorers s5-var s34-o1 s34-o2 s34-o3 s18-p1 s18-p2 --field scores_o
  python scripts/seedlm32_score.py --set $set --scorers s5-var s34-k1 s34-k2 s34-k3 s18-p1 s18-p2 --field scores_k
  for f in k o; do
    python scripts/seedlm32_select.py --set $set --rule ensemble --field scores_$f --output reports/seedlm/selection34_${set}_$f.json
  done
done
python scripts/seedlm34_diagnose.py --fit 120 --output reports/seedlm/seedlm34_diagnosis.json
```

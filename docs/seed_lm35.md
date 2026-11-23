# SeedLM-35: pyx 검토기 — 설명이 프로그램에서 나올 확률로 다시 매기기

작성일: 2026-11-23. 측정 전에 등록했다(`reports/seedlm/seedlm35_protocol.json`).

## 요약

- [SeedLM-34](seed_lm34.md)에서 같은 뼈대의 후보들은 고정 토큰을 공유하고 구멍만 달라, 코더가 구조를 좋아하면 맞는 것과 틀린 것이 함께 올랐다. 둘을 가르는 질문은 거꾸로 된 방향, "이 프로그램이 설명의 단어를 다 설명하는가", 즉 P(설명 | 프로그램, 예시)다(coder-reviewer 재정렬, Zhang 외 2022).
- 코더와 같은 크기(1,240만 파라미터), 같은 어휘의 검토기 r35-a를 처음부터 40분 학습했다. 코더의 학습 출처를 거꾸로 읽고, 바꿔 쓴 문장 2,400개와 뼈대 과제 772개를 더했다. 보류 교사 과제, 개발 과제, 검증·시험은 쓰지 않았다.
- 교사 보류 과제에서만 체크포인트와 결합 가중치를 골랐다(10분 체크포인트, 여섯 코더 평균 + 1 × 검토기). 후보만으로 64 → 67이었지만 20가지 설정 중 최고라 낙관적인 수다.
- **개발 과제(주 측정): 시스템 65**, SeedLM-32와 같다. 지지되지 않았고 채택할 것이 없다. 후보만으로는 58(SeedLM-32 59).

## 결과

| | 교사 보류 139, 후보만 | 교사 보류, 시스템 | 개발 93, 후보만 | 개발, 시스템 |
|---|---|---|---|---|
| 여섯 코더 평균 (SeedLM-32) | 64 | 58 | 59 | 65 |
| 검토기 혼자 (체크포인트 넷) | 43~49 | 48~51 | | |
| **여섯 코더 평균 + 검토기 (10분, 가중치 1)** | **67** | 59 | **58** | **65** |

`reports/seedlm/review35_teacher.json`, `review35_development.json`.

검토기 혼자는 코더보다 훨씬 못 고른다. 이 크기의 모델이 설명 전체의 확률을 매길 때는 설명의 문체와 길이가 프로그램보다 크게 작용한다. 코더 확률에 더하면 교사 보류 과제 셋이 바뀌었는데, 20가지 설정에서 고른 최댓값의 잡음 크기이고, 개발 과제에서는 하나도 바뀌지 않았다. [SeedLM-36](seed_lm36.md)의 엄격한 판정으로 보면 개발 과제 시스템은 55로, 지금 시스템(58)보다 낮다.

## 재현

```bash
python scripts/seedlm35_review_train.py --name r35-a --minutes 40 --save-every 10 --synthetic 150000 \
  --shares "real=0.08,teacher=0.15,variant=0.08,oneliner=0.01,paraphrase=0.12,sketch=0.06"
python scripts/seedlm35_review_select.py score --set teacher --reviewers r35-a-m10 r35-a-m20 r35-a-m30 r35-a
python scripts/seedlm35_review_select.py explore --set teacher --output reports/seedlm/review35_teacher.json
python scripts/seedlm35_review_select.py score --set development --reviewers r35-a-m10
python scripts/seedlm35_review_select.py measure --set development --reviewer r35-a-m10 --rule coder_reviewer --lam 1 \
  --output reports/seedlm/review35_development.json
```

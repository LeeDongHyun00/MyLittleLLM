# 현재 상태 — 2026-11-05

계약 버전 **1.1.0**. 이 파일은 현재 상태의 요약이며 과거 실험 원본을 대체하지 않는다.

## 어디서 이어갈 것인가

- 저장소: [LeeDongHyun00/MyLittleLLM](https://github.com/LeeDongHyun00/MyLittleLLM).
- 새 세션 시작 브랜치: `docs/project-contract-and-session-roles`.
- [통합 PR #4](https://github.com/LeeDongHyun00/MyLittleLLM/pull/4)는 2026-11-05 기준 브랜치에 병합했다.
  병합 커밋: `04ff35a5143ecba349fbbb82ed713f5aba9221da`. E01/D01, R01/L01 결과와 세션 CI를 포함한다.
  원래 작업 브랜치 `s0/i01-session-collaboration`과 역할별 커밋은 이력으로 보존한다.
  [협업 운영](COLLABORATION.md), [세션 상태](../../contracts/sessions.json), [협업 이슈 #3](https://github.com/LeeDongHyun00/MyLittleLLM/issues/3).
- 현재 구현 기준 커밋: `04ff35a5143ecba349fbbb82ed713f5aba9221da`.
  기존 기능 기준 `d0ac404c8cda106e703bd04a691aaf8e10abf056`과 역할별 개발 이력을 상속한다.
- 부모 브랜치 `feat/raft-step-opsd-compression`의 [draft PR #1](https://github.com/LeeDongHyun00/MyLittleLLM/pull/1)은 아직 병합 전이다.
  `main`은 `83d9d9912cdc74af0363efd4717166fcfc2224ca`로 현재 구현보다 이전 상태다.
- 기준 브랜치는 부모 기능 PR 위에 있다. [PR #2](https://github.com/LeeDongHyun00/MyLittleLLM/pull/2)와 PR #1은 아직 draft이며 `main` 병합은 별도다.
- 기존 파일/모델이 다른 세션에 자동으로 존재한다고 가정하지 않는다. Git 제외 모델·원문·색인은 새 환경에서 재확인한다.

## 동작과 자원

공식 원문 색인 8개·1,295구간, 부분 규칙 9개, 조건부 비용 계산기, 한국어 조건 문답이 있다.
입시 전체 범위는 서울대·부산대·경희대 2025/2026 수시이며 규칙 범위는 더 좁다.
총 URL 레코드 26개 중 18개 다운로드 성공, 8개 실패. 다운로드와 정답 승인은 다르다.

`--pipeline grounded`를 명시해 사용한다. CLI 기본값은 `legacy`, `--claim-check`는 선택 옵션이다.
규칙 선택과 개인 조건은 명시 입력이다. 자연어 자동 추출·모든 의미 충돌 감지·전국 대학 상담은 구현하지 않았다.

기존 설치는 CPU 4개·메모리 16GiB·GPU 없음, llama.cpp b11960, 8,192 문맥, 슬롯 1이다.
모델 경로는 `models/Qwen3-4B-Q4_K_M.gguf`, API는 해당 환경 내부 `http://127.0.0.1:8080/v1`이다.
**2026-11-04 점검에서 `/health`는 connection refused였다.** 과거 실행 성공을 현재 서버 가동으로 해석하지 않는다.
11월 5일 통합에서는 서버 상태 재점검·재시작·모델 추론·학습을 수행하지 않았다. 실행 세션은 [설치/실행 문서](../running.md)를 확인한다.

## 증거 장부

아래는 2026-11-02 고정 실행의 자동 지표이며 서로 다른 표본을 합산하지 않는다.

| 실험 | 관측 결과 | 해석/근거 |
|---|---|---|
| 규칙 확대 12개 | 같은 근거 Qwen 5/12, 규칙·보류 11/12; 실제 Qwen 호출 12회, 판단 24건 | [요약](../../reports/grounding_v2_summary.json), [24건 의미 검토](../../reports/grounding_v2_qualitative_review.json). AI 작성 조건/gold, 독립 사람 검수 없음 |
| 규칙/검증 18개 | Qwen 13/18, 규칙·보류 18/18, Qwen+검증 6/18, 전체 결합 15/18 | [요약](../../reports/grounding_summary.json). 검증 필터가 개선을 보장하지 않음 |
| 전처리 12개 | concise 10/12, 입력 확대 10/12, 전처리 9/12 | [결과](../../reports/preprocessing_report.md). 전처리 정확도 개선 미확인 |
| 학습 28개 객관식 | 기본 22/28, SFT 21/28, OPSD 21/28 | [결과](../../reports/holdout_report.md). 소량 학습의 일반화 개선 미확인 |
| 압축 보충 비교 | 문맥 토큰 1.93% 절감 | [결과](../../reports/holdout_report.md). 제한된 비교이며 일반 성능 향상 근거 아님 |
| 코드 검사 | 74개 통과(이전 기능 작업 기록) | [구현/검증](../guided_grounding.md). 이번 문서 작업에서 모델 평가를 다시 실행한 결과가 아님 |

12개 실행 고정 커밋은 `cc99c19ac6b663a0428dc8820c24094d7c1d9fc7`,
18개는 `829f712af32875ada9e825de7ab0a718a624710d`다. 현재 코드로 과거 잠금 실행을 덮어쓰지 않는다.
48/12/40 정식 분할 실험과 독립 전문가 검수는 미완료다.

## 해결해야 할 것

- **V12 검색:** 일반 주거비 월 상한 질문에 BM25가 PDF 23·3쪽을 제공해 상한 조항 PDF 10쪽이 빠졌다.
- **V12 생성:** 1,315 입력 토큰으로 문맥 초과는 아니었다. 예시 문구/특수 토큰 모양 문자열 반복 후 384토큰에서 잘렸다.
  입력 근거 누락과 생성 실패를 따로 실험해야 한다. 무조건 토큰을 늘리는 처방의 효과는 확인되지 않았다.
- **검수:** [독립 검수 양식](../../data/review/grounding_v2_blind_review.csv)의 검수자·답·날짜는 비어 있다.
  기존 공개 사례에 사람 검수를 추가해도 새 대학/새 규칙 독립 시험으로 소급 변경할 수 없다.
- **채점:** V04/V07의 약한 조건부 답변이 정규식에서 통과했다. 자동 점수와 의미 평가를 함께 유지한다.

## 이번 협업의 상태

- **E01 준비 완료:** [프로토콜](../evaluation/e01_protocol.md), 접근/누출 manifest와 H1 검수 양식을 준비했다.
  새 독립 후보·H1 승인·최종 잠금은 아직 없다.
- **D01 부분 완료:** [원문 진단](../../reports/d01_source_report.md). 별도 동서대 배포본의 38청크에서
  상한/각주는 존재하지만 V12의 상한 근거가 3위여서 k=2에 빠짐을 확인했다.
  기존 제주대 해시 원문 재확보와 공식 최신 수정본 대조는 외부 TLS 다운로드 실패로 남았다.
- **R01 부분 완료, 후보 미채택:** [검색 개발 보고](../../reports/r01_retrieval_development.md).
  문장 가중치는 개선 없음. 조항 경계 후보는 V12를 찾지만 완전 근거 묶음 8/11→6/11로 퇴보했다.
  두 후보는 별도 개발 모듈로만 보존하고 기본 BM25 k=2는 유지했다.
- **L01 개발 구현 완료:** [경계·설명 보고](../../reports/l01_rule_boundaries.md).
  입증된 false 필요 조건만 설명한다. unknown·모순·정수 제약·부분 판정과 원문 해시 연결을 유지했다.
  선택적 설정 확장을 계약 1.1.0에 반영했다. H1 검수와 실서비스 지연 측정은 미완료다.
- **통합 검증:** 관련 회귀 검사 86개 통과. 원문/모델/최종 시험 없이 실행했고 기존 동결 파일 13개는 보존했다.
- **이벤트 연결:** GitHub Actions push/PR에서 계약·세션 인수인계와 관련 회귀를 검사한다.
  11월 5일 기준 브랜치 병합 커밋의 [세션 CI](https://github.com/LeeDongHyun00/MyLittleLLM/actions/runs/41973487999)와
  [계약 검사](https://github.com/LeeDongHyun00/MyLittleLLM/actions/runs/41973487948)가 성공했다. 실행 커밋/URL은 최신 S0 인수인계에 기록한다.
  외부 webhook은 관리 권한/수신처가 없어 미연결이다.
- **평가 경계:** 이번 공개 개발 파일은 [노출 추가 기록](../../data/evaluation/e01/exposure_addendum_20261104_i01.json)에 연결했다.
  E01의 최초 스냅샷을 바꾸지 않았고 E02 전 전체 노출 목록과 그룹을 다시 점검해야 한다.

## 다음 순서

1. S1 `D01`: 기존 원문과 공식 최신 수정본의 대조를 마친다. 별도 배포본의 개발 결과와 구분한다.
2. S2 `R01`: 새 계획/run에서 검색 단위와 반환 문맥·인접 예외 연결을 검토하고 이번 퇴보 사례도 함께 비교한다.
3. S4 `M01`: 고정된 근거를 주고 생성 반복·잘림 처리를 비교한다.
4. S5 `E02`: H1 두 명/보관자·새 그룹·사전 동결이 준비되면 별도 평가한다.
5. S0 `I01`: 위 연구 결과가 준비되면 후속 통합과 기준 브랜치 갱신을 수행한다.

카드별 준비·진행·완료 상태는 [TASKS](TASKS.md), 커밋별 세션 기록은 [sessions.json](../../contracts/sessions.json)을 확인한다.

# 설치 및 실행

2026-11-02에 이 대화의 클라우드 환경에 직접 설치한 CPU 프로토타입이다. 서버는 `127.0.0.1:8080`에서 실행하며 외부에 공개하지 않는다. 사용자 PC의 localhost와는 다르다.

## 설치 기록

- Linux x86_64, CPU 할당 4코어, 메모리 제한 16GiB. GPU는 발견되지 않았다.
- 모델: 공식 `Qwen/Qwen3-4B-GGUF`의 `Qwen3-4B-Q4_K_M.gguf`, 2,497,280,256바이트.
- llama.cpp: `b11960`, 실제 버전 출력 `0.8.0-dev`, commit `8642541b2`.
- 다운로드 revision과 SHA-256은 [runtime.lock.json](../config/runtime.lock.json)에 고정했다. 두 파일의 체크섬을 실제 대조했다.
- 컨텍스트 8,192, CPU 스레드 4, 동시 처리 슬롯 1, thinking 비활성화, 추론 temperature 0, seed 42.
- 가중치를 업데이트하는 미세조정은 수행하지 않았다. 설치된 GGUF로 실제 추론과 RAG 비교를 실행한다.

새 Linux x86_64 환경에서:

```bash
python -m pip install -r requirements-data.txt
python scripts/install_runtime.py
python scripts/collect_sources.py
python scripts/build_index.py
bash serve.sh
```

`serve.sh`는 포그라운드에서 실행한다. 별도 터미널에서 다음을 실행한다. 이미 서버가 실행 중이면 다시 띄울 필요가 없다.

```bash
python scripts/probe_model.py
python chat.py --year 2026 --institution 경희대학교 \
  --question '지역균형전형의 학교별 추천 인원에 제한이 있나요?'
```

한 번에 질문을 주지 않으면 선택한 학년도·대학 범위에서 대화한다. 현재 구현은 매 질문을 독립 처리하며 이전 대화의 조건을 자동으로 기억하지 않는다.

```bash
python chat.py --year 2026 --institution 서울대학교
```

## 우선순위 1: 입시 상담

지원 대학/연도 → 해당 원문만 검색 → 근거와 질문을 Qwen에 제공 → JSON 형식과 인용 ID 확인 → 답변·원문 URL·PDF 페이지 반환.

색인은 [corpus.json](../config/corpus.json)의 공식 규정 8개만 읽는다. 입시 모집요강 6개, 2025 교육부 국가장학금 기본계획, 2026년 1학기 주거안정 시행계획의 PDF 1–25쪽이다. 학생부 안내서, FAQ 구간, QA 답안은 색인하지 않는다. 정책과 FAQ가 한 파일인 주거 자료는 페이지 범위로 분리했다.

검색은 외부 임베딩 API 없이 한국어 문자 n-gram BM25를 사용한다. 900자 구간, 160자 중첩이며 상위 3개 서로 다른 페이지의 구간을 제공한다. 임의 질문에 대해 필요한 근거가 항상 검색된다는 보장은 없다.

최초 실행에서 연도 비교표를 잘못 읽는 오류가 발견되어, 여러 학년도와 변경 표현이 함께 있는 페이지 12개를 임시 보류했다. 이후 색인은 1,295개 구간이다. 이 휴리스틱은 보수적이라 유효한 조항도 보류할 수 있다. `outputs/index_review.json`의 페이지를 표 구조와 함께 검수해 다시 포함해야 한다. 현재 상태는 전국 입시 상담 서비스의 완성본이 아니다.

형식과 인용 ID 검사는 실제 문장과 근거의 의미 일치까지 보장하지 않는다. 근거가 있어도 모델이 오독할 수 있으며 결과는 연구용이다.

## 우선순위 2: 장학·주거 안내 및 계산

주거 규정 검색:

```bash
python chat.py --year 2026 --task housing --semester 1 \
  --question '주거안정장학금의 월 지원 한도는 얼마인가요?'
```

가상의 승인 조건을 명시한 계산과 설명:

```bash
python chat.py --year 2026 --task housing --semester 1 \
  --facts data/examples/housing_scenario.json \
  --question '4월 인정 주거비가 12만원일 때 받을 금액을 설명해 주세요.'
```

`--facts`는 사용자 입력 조건이다. 모델이 자유롭게 생성한 값을 승인된 사실로 사용하지 않는다. 계산기는 2026년 1학기 3–6월만 지원하고, 선발·지급 승인·중복지원 없음이 확인되지 않으면 계산을 거부한다. 첫 달 정액/이후 실비 상한은 원문 PDF 10쪽에 근거한다. 해당 조항 구간을 추가 조회한 경우 `calculator_rule_lookup`으로 로그를 남겨 일반 검색과 구분한다.

LLM을 거치지 않는 계산도 가능하다.

```bash
python scripts/calculate_cost.py housing data/examples/housing_scenario.json
python scripts/calculate_cost.py net data/examples/cost_scenario.json
```

제공된 예시에서는 주거비 지원 12만원, 한 학기 순부담 300만원, 대출 후 필요 현금 200만원, 대출 원금 100만원이 나온다. 대학별 실제 요금이나 개인의 실제 지원액을 의미하지 않는다. 대출은 장학금처럼 순부담에서 빼지 않는다.

## 실험 재현

```bash
python -m unittest discover -s tests -v
python scripts/run_pilot.py --allow-unreviewed --output outputs/pilot-new.jsonl
```

이 명령은 공개된 미검수 개발 예시 5개로 문서 없는 모델과 RAG를 비교한다. 정답·근거 정답 페이지·정답 필드는 모델에 전달하지 않는다. 최종 시험 정확도나 60:40 학습 실험이 아니다. 기존 출력 파일은 덮어쓰지 않는다. 데이터 분할·인간 검수·LoRA 비교는 [실험 설계](experiment_design.md)에 따라 별도로 진행해야 한다.

원문·색인·가중치·서버 바이너리는 Git에서 제외한다. 새로운 환경에서 수집한 문서가 기존 스냅샷 해시와 다르면 임의로 동일 판본이라고 처리하지 말고 수정 여부를 검토한다.

# MyLittleLLM

Qwen3-4B를 활용해 한국 대학 입시·장학 정보 접근 격차를 줄일 수 있는지 검증하는 프로젝트.

**우선순위 1: 근거를 제시하는 입시 상담. 우선순위 2: 장학금·주거·실부담 비용 안내.** 공식 문서 검색(RAG), 검증 가능한 조건·계산 코드, LLM의 설명을 결합한다.

## 문서

- [전체 우선순위와 구현 범위](docs/priorities.md)
- [60:40 실험에 대한 비판과 수정 설계](docs/experiment_design.md)
- [2025·2026학년도 모집요강·공식 해설 및 장학 자료](docs/data_sources.md)
- [모델 경로 및 실행 환경 확인 결과](docs/environment.md)
- [직접 설치와 RAG·장학금 계산 실행](docs/running.md)
- [실제 예비 실행 결과와 발견한 오류](reports/pilot_report.md)
- [RAFT·설명 감독학습·OPSD·LLMLingua-2 실행](docs/distillation.md)
- [증류·압축 개발 실험 결과](reports/distillation_report.md)

## 2026-11-02 현재 상태

- 서울대·부산대·경희대의 2025·2026학년도 **수시 모집요강 6개 + 공식 안내서 6개** 수집. 서울대 안내서는 교육청 학교 홈페이지 배포본을 사용했다.
- 장학금 자료를 포함한 URL 레코드 26개 중 18개 다운로드 성공: PDF 14개(1,075페이지), HWPX 1개, HTML 3개. 8개 접근 실패도 목록에 기록했다. 게시물/첨부가 별도 레코드이므로 고유 문서 수와 다를 수 있다.
- 수집기, 페이지별 PDF 추출, SHA-256 스냅샷, 그룹 단위 분할 코드, QA 초안 5개 작성.
- **후속 작업에서 Qwen3-4B Q4_K_M과 llama.cpp를 직접 설치하고 CPU 서버 및 RAG CLI를 실행했다.** 공식 원문 8개에서 비교표 검수 보류 후 1,295개 검색 구간을 사용한다.
- 주거비 지원과 실부담 비용 계산기를 구현했다. 지원 조건과 기간은 명시적으로 확인하며 대출을 장학금으로 차감하지 않는다.
- 공개 개발 예시 5개에 대해 문서 없는 모델/RAG 10회 추론을 완료했다. RAG에서도 전년도 규정 오독과 지급액 오류를 관찰했으며 실패를 포함해 결과를 기록했다.
- 후속 개발 실험에서 **RAFT·답변/설명 지도학습 10단계 + OPSD 3단계**를 CPU에서 실행하고 일반 LoRA 어댑터를 저장했다. QLoRA는 사용하지 않았다. 공식 LLMLingua-2 실행과 조항 보호 압축 옵션도 추가했다.
- **사람이 승인한 gold 데이터와 잠근 시험셋의 정확도 측정 결과는 아직 없다.** 공개 예시는 전부 `pending`이고 최종 시험셋으로 재사용하지 않는다. 미세조정도 5개 개발 예시의 작동 시험이며, 최종 답변에서 일부 조건 누락을 관찰했다.

## 제안한 학습 방식의 수정

모집요강은 근거 문서이고 해설은 정답 후보 자료다. 질문마다 적용 학년도·전형·조건·예외·근거 페이지를 연결하고 검수해야 정답셋이 된다.

60:40은 **개발 60% / 잠근 시험 40%**로 유지하되, 개발용을 나누어 전체 기준 **학습 48% / 검증 12% / 시험 40%**로 시작한다. 같은 규칙의 연도별 버전·바꿔 쓴 질문·문서 복사본은 한 그룹으로 묶는다. 그룹 크기 때문에 실제 문항 비율은 달라질 수 있다.

현재 GGUF로 RAG 기준선을 구현하고 개발 예시를 실행했다. BF16 Qwen3-4B의 소규모 LoRA·OPSD 작동 시험도 완료했으며, 정식 효과 비교에는 검수된 별도 시험셋이 필요하다. 미세조정으로 매년 바뀌는 요강을 암기시키는 것만으로는 최신 상담 정확도를 보장할 수 없다.

## 데이터 수집

Python 3.10 이상. PDF 추출에는 pypdf를 사용한다.

```bash
python -m pip install -r requirements-data.txt
python scripts/collect_sources.py
```

일부 자료만 다시 받을 때:

```bash
python scripts/collect_sources.py --ids snu-2026-susi khu-2026-guide --output outputs/retry.jsonl
```

- 입력 목록: `data/sources.jsonl`
- 이번 수집 결과: `data/retrieval_snapshot.jsonl`
- 원문: `data/raw/` (Git 제외)
- PDF 페이지별 추출: `data/extracted/` (Git 제외)
- 새 실행 결과: `outputs/retrieval.jsonl` (Git 제외)

표·각주의 자동 추출 오류는 별도 검수한다. 부산대 2025 가이드북 31페이지는 OCR이 필요하다. HWPX와 HTML은 원본을 수집하며 현재 수집기에서 본문을 추출하지 않는다. 파일의 다운로드 성공은 정답 승인이나 학습 이용 허가를 의미하지 않는다.

## 검수 후 분할

```bash
python scripts/split_dataset.py data/private/reviewed_qa.jsonl --output outputs/frozen-v1
```

`review_status=approved`인 QA만 받는다. 실제 검수자·근거를 남긴 후 사용한다. 입력에 동일 질문이 다른 그룹으로 나타나면 중단하고, 이미 존재하는 출력 폴더에는 시험셋을 덮어쓰지 않는다. 의미가 같은 규칙의 자동 탐지는 구현하지 않았으므로 `leakage_group` 검수는 필수다.

새 대학 일반화 실험은 대학 수를 확대한 뒤 `--group-key institution_group`으로 별도로 분할한다. 연도 갱신 실험은 이 그룹 분할과 다른 평가 프로토콜이다.

## 코드 검증과 모델 연결 확인

```bash
python -m unittest discover -s tests -v
python scripts/probe_model.py --base-url http://127.0.0.1:8080/v1
```

테스트는 그룹 누출 방지, 미검수 답안 거부, 범위별 검색, 인용 ID 검증, 주거비 상한과 첫 달 예외, 대출과 비용의 구분을 검사한다. 코드 테스트 통과는 모델 답변의 정확도를 보장하지 않는다.

원문 저작물은 저장소에 재배포하지 않는다. URL·해시·수집 상태와 자체 작성한 문서/QA 초안만 공개한다. 원문별 이용 조건은 목록의 `license_note`에 기록한다.

# RAFT·설명 감독학습·OPSD·LLMLingua-2 실행

사용자의 변경 요청에 따라 QLoRA는 제외했다. 최신 증류 연구에서는 별도의 대형 교사 모델을 요구하지 않는 **Self-Distilled Reasoner / OPSD (ICML 2026)**를 선택했다. 기본 Qwen3-4B는 BF16 그대로 동결하고 일반 LoRA만 학습한다. 4비트 양자화 기반 학습은 하지 않는다.

## 논문과 구현의 대응

| 연구 | 이번 구현 | 재현 범위와 차이 |
|---|---|---|
| [RAFT (2024)](https://arxiv.org/abs/2403.10131) | 올바른 연도·대학의 근거와 다른 연도/대학의 방해 문서를 함께 제시하고 출처 ID를 출력 | 정답 근거가 항상 포함되는 개발용 oracle-context. 실제 검색 성능 평가 및 원논문의 혼합 비율 재현은 아님 |
| [Distilling Step-by-Step (ACL 2023)](https://aclanthology.org/2023.findings-acl.507/) | 같은 질문에 답변 task와 짧은 근거 설명 task를 별도 지도학습 | Codex가 원문에 대조해 작성한 설명 후보 5개. 별도 외부 교사 API나 원논문의 T5 다중과제 실험은 사용하지 않음 |
| [Self-Distilled Reasoner (ICML 2026)](https://proceedings.mlr.press/v306/zhao26be.html) | 학생의 실제 샘플 생성 → 정답 설명을 추가로 본 동결 교사의 전체 어휘 확률 → JSD로 학생 LoRA 업데이트 | 단일 CPU, rank 4, 생성 최대 24토큰의 작동 시험. 논문의 GPU·수학 데이터·학습 규모를 재현하지 않음 |
| [LLMLingua-2 (ACL 2024)](https://aclanthology.org/2024.findings-acl.57/) | 공식 다국어 BERT 압축 모델 실행 + 규정 조항 보존 + 상담 CLI 선택 옵션 | 한국어 입시 성능은 별도 검증 필요. Qwen 토크나이저로 길이를 측정하며 안전한 조항 보존 때문에 압축률이 낮을 수 있음 |

OPSD의 교사는 **학습 시작 시점의 기본 모델**로 고정한다. 학생은 업데이트된 LoRA를 사용하며, 교사 평가 시 어댑터를 비활성화한다. 교사에게만 `reference_solution`을 전달한다. 학생의 생성 토큰 위치에 맞추어 두 분포를 정렬하고 전체 어휘에 대한 대칭 JSD를 최소화한다. 교사에는 gradient가 전달되지 않는다. SFT 후에도 교사를 학생으로 바꾸지 않는다.

모든 프롬프트에서 Qwen의 `enable_thinking=False`를 사용한다. OPSD 학생 샘플은 temperature 1, top-p 1, top-k 0이며, 개발 답변 비교는 greedy 생성·최대 192토큰이다. 짧은 근거 설명을 학습하는 이번 설정은 원논문의 긴 수학 풀이 실험과 다르다.

두 SFT task는 번갈아 학습한다. 시스템·질문·근거 입력은 loss에서 마스킹하고 정답 출력과 EOS에만 loss를 적용한다. 입력이 설정한 길이를 넘으면 조항이나 정답을 자르지 않고 실패시킨다.

## 환경 설치

```bash
cd /workspace/MyLittleLLM
python -m venv .venv
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python scripts/install_training_assets.py
```

GPU에서는 해당 GPU에 맞는 PyTorch 빌드를 설치하고 학습 명령에 `--device cuda`를 추가한다. CPU에서도 실행 가능하지만 소규모 작동 시험에 적합하다. 현재 환경은 CPU 4코어·16GiB이고, 기존 llama.cpp 서버와 BF16 학습을 동시에 실행하면 메모리가 부족할 수 있다. 학습 시간 동안 서버를 중지하고 종료 후 `./serve.sh`로 복구한다.

설치기는 `config/distillation.json`의 커밋으로 Qwen3-4B와 압축 모델을 고정하고 파일 해시를 `outputs/training_assets.json`에 남긴다. 원본 GGUF는 덮어쓰지 않는다. 가중치, 원문 인용을 포함한 생성 데이터, 어댑터는 Git에서 제외한다.

## 개발 데이터와 학습

```bash
python scripts/prepare_distillation.py \
  --output outputs/distillation-smoke-v1 \
  --allow-unreviewed-smoke

.venv/bin/python scripts/train_distillation.py \
  --examples outputs/distillation-smoke-v1/examples.jsonl \
  --output outputs/raft-step-opsd-smoke-v1 \
  --sft-steps 10 --opsd-steps 3 \
  --allow-unreviewed-smoke --run-probes
```

현재 5개 질문은 독립 규칙 그룹 3개에 속한다. 2025·2026년 같은 규칙의 버전은 독립 표본이 아니다. 이 공개 예시는 기존 개발 실험에서 이미 사용됐으며, 새로운 시험셋으로 선언하거나 60:40 분할하지 않는다.

`--allow-unreviewed-smoke`는 미검수 데이터를 명시적으로 사용하는 작동 시험 옵션이다. `pending`을 `approved`로 바꾸지 않는다. 정식 데이터 준비에서는 이 옵션을 빼고, QA와 설명에 모두 독립 검수 상태·검수자 정보를 제공해야 한다. 공식 원문 해시, 추출문 provenance, 학년도, 인용 범위를 확인하며 이미 존재하는 실험 폴더에는 덮어쓰지 않는다.

정식 실험에서는 기존 `scripts/split_dataset.py`로 검수된 규칙 그룹을 먼저 분리하고 **train 부분만** 학습 데이터 준비에 넣는다. 잠근 test의 정답·설명·QA를 학습용 방해 자료로 추가하지 않는다. 현재 학습기는 별도의 검증/조기 종료/최종 시험 평가를 구현하지 않은 실험용 코드다.

출력:

- `steps.jsonl`: 실제 SFT loss, OPSD JSD, gradient norm, 샘플 생성 기록.
- `sft_adapter/`: RAFT·답변/설명 지도학습 후 어댑터.
- `final_adapter/`: OPSD 후 어댑터와 토크나이저.
- `development_probes.jsonl`: 동일한 공개 개발 질문의 학습 전·SFT 후·OPSD 후 생성문. 정확도 평가가 아님.
- `run_summary.json`: 설정, 버전, 학습 시간, 메모리, 어댑터 가중치 변화량. 완료한 경우에만 생성.

OPSD의 24토큰 상한에서는 JSON 답변이 완성되지 않을 수 있다. 이 경우에도 실제 학생 생성 경로의 토큰 분포로 역전파가 가능한지 확인할 수 있지만, 최종 상담 능력이 학습됐다는 증거가 되지 않는다. 정식 실험은 완결된 답변을 포함하는 생성 길이를 검증셋에서 정해야 한다.

## LLMLingua-2 측정과 상담 연결

```bash
.venv/bin/python scripts/run_compression.py \
  --examples outputs/distillation-smoke-v1/examples.jsonl \
  --output outputs/compression-smoke-v1 --rate 0.5

.venv/bin/python chat.py \
  --year 2026 --institution 경희대학교 --compress-context \
  --question '2026학년도 지역균형전형의 학교별 추천 인원 제한은 무엇인가요?'
```

위 상담 명령은 실행 중인 기존 GGUF 서버를 사용한다. 학습된 어댑터를 자동 배포하는 명령은 아니다. 학습 출력 형식과 현재 상담 JSON 형식도 다르므로, 어댑터를 서비스에 쓰려면 출력 규약을 통일하고 별도 평가해야 한다.

일반 압축은 연구 비교용 파일에만 남긴다. 상담 옵션은 숫자·부정·예외 등이 있는 조항을 통째로 보존하고 나머지 부분에만 실제 LLMLingua-2를 적용한다. 결과가 비거나 토큰 수가 늘면 원문으로 복귀한다. 문서 ID·연도·페이지·출처 해시는 유지하고 원래 검색 색인은 수정하지 않는다. 정답이나 oracle ID로 보호 조항을 결정하지 않는다.

LLMLingua 0.2.2의 초기화는 압축 모델과 무관한 GPT 토크나이저 다운로드를 요구한다. 이 환경에서는 해당 호스트 요청이 거부됐다. `load_compressor()`는 네트워크 우회 없이 **이미 설치된 Qwen 토크나이저를 길이 계산에 사용**하고, 공식 LLMLingua-2 분류 모델과 압축 로직을 그대로 호출한다. 일반 압축의 길이도 Qwen 기준이다.

보호 검사는 문장 보존 검사이며 모든 의미를 검증하는 장치가 아니다. 문장 분리·PDF 표 추출·문서 버전 오류는 별도의 문제다. `compression` 실행 로그에서 전후 토큰 수와 처리 시간을 확인한다. 토큰 감소를 실제 전체 응답 속도 향상으로 간주하지 않는다.

## 검증

```bash
.venv/bin/python -m unittest discover -s tests -v
```

교사 정답의 학생 입력 누출, 입력 loss 마스킹, 문맥 초과 시 중단, JSD의 교사 gradient 차단 및 학생 업데이트 방향, 조항 보존과 출처 ID 불변성을 검사한다. 실제 실행 결과는 [실험 보고서](../reports/distillation_report.md)에 기록한다.

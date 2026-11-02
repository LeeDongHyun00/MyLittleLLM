# 학습과 분리한 시험셋 v1

이 시험은 이전 학습 질문 5개를 바꿔 쓰는 방식이 아니다. 부산대학교 2025·2026 수시 규정 및 교육부 2025년 2월 국가장학금 계획의 **다른 규칙**으로 새 질문 28개를 작성했다. 학습의 추천 인원·주거비 월 지급액 규칙과 근거 페이지를 제외했다. 부산대는 어댑터 학습에 등장하지 않은 대학이다.

정답은 Codex가 공식 원문과 대조한 연구용 기준이며 독립적인 사람 검수는 받지 않았다. 따라서 학습과 분리된 실측 평가이지만, 전문가 검수와 대표 표집을 마친 최종 벤치마크는 아니다. 기반 모델의 사전학습 노출 여부도 증명할 수 없다.

범위는 지원 절차·예외·연도 변경·장학 제도 안내다. 수능 최저의 복잡한 조합, 대학별 내신 환산, 합격 가능성 예측 전반을 검증한 시험은 아니다.

시험 질문·선택지·정답·근거 페이지는 [검토용 CSV](../data/evaluation/holdout_v1.csv)와 [JSONL](../data/evaluation/holdout_v1.jsonl)로 제공한다. [프로토콜](../data/evaluation/holdout_v1_protocol.json)에 원문 버전, 입력 및 정답 해시, 모델 해시가 있다. 원문 발췌를 포함한 실제 요청은 로컬 `outputs/evaluation-run-v1/requests.jsonl`에 보존한다.

## 고정된 평가 설계

- 객관식 28문항, 규칙 그룹 15개. 정답 위치 A/B/C/D를 7개씩 배치했다.
- 그중 8문항을 선택지 없는 서술형으로도 평가한다. **총 36개의 독립 질문이라고 세지 않는다.**
- 세 모델 조건마다 28 + 8회, 총 108회 생성한다. 모델 순서는 문항별로 미리 무작위화했다.
- 기본 Qwen3-4B Q4_K_M, SFT 어댑터, OPSD 최종 어댑터를 **같은 양자화 기반과 서버 설정**에서 비교한다.
- SFT와 OPSD를 동시에 적용하지 않는다. 기본 모델은 두 어댑터 scale=0, SFT는 id=0만 1, OPSD는 id=1만 1이다. 매 요청 전 서버 상태를 확인한다.
- LoRA를 공식 llama.cpp b11960 변환기로 f32 GGUF로 변환했고, 두 어댑터의 144개 텐서가 원본과 값·형상이 모두 일치함을 확인했다.
- 학습은 BF16에서 수행했지만 이번 결과는 Q4 배포 조건이다. BF16 자체의 평가 결과로 읽으면 안 된다.

객관식은 정답 근거와 관련 문서 발췌를 직접 제공하는 **oracle-context** 시험이다. 연도가 다른 문서를 함께 주는 사례도 있다. A/B/C/D 한 글자로 출력을 제한하므로 출력 형식 준수나 자유로운 상담 능력의 점수가 아니다. 자료가 없는 학년도, 개인 합격 확률, 학년도 미지정 문항도 포함한다.

서술형은 기존 BM25 검색기로 **top-2 구간**을 검색하여 모든 모델에 똑같이 제공한다. 학년도가 없는 문항에서는 임의의 연도 문서를 검색하지 않는다. 정답·선택지·해설은 서술형 입력에 넣지 않는다. JSON 문법/스키마를 강제로 제한하지 않고 status, answer, citations, follow_up을 요청한다. 동일한 160토큰 상한에서 잘리거나 형식이 틀리면 실패로 기록한다. 이는 긴 출력을 허용한 전체 성능과 다를 수 있다.

원문 자체를 읽을 수 있는 것은 open-book 평가의 조건이다. 정답 문구나 시험 QA를 검색 색인에 추가한 것은 아니다. 원문 버전·색인 해시·학습 데이터 해시·모델 파일 해시·시험 입력 및 정답 해시를 보존한다.

## 채점

객관식은 선택 문자와 고정 정답을 비교한다. 문항 단위 정답률, 분야별 정답률, 같은 문항의 개선/퇴보 수, 규칙 그룹 단위 bootstrap 95% 구간을 보고한다. 질문 두 개가 같은 규칙에 속하면 독립 사례로 취급하지 않는다. 작은 그룹 수에서는 bootstrap 구간도 불안정하며, 전부 맞은 경우 퇴화한 구간이 나올 수 있다. 전국 대학 상담 정확도로 일반화하지 않는다.

서술형은 아래를 분리한다.

1. JSON/필드 형식 및 생성 종료 여부.
2. 답변/보류 상태의 적절성.
3. 해당 연도 근거 페이지의 인용 여부. 페이지 일치는 의미적 뒷받침 전체를 증명하지 않는다.
4. 사전에 고정한 필수 조건·금지 주장 패턴 충족 여부.

자동 서술형 점수는 **엄격한 어휘 기반 기준 통과율**이다. 의미가 맞아도 표현이 다르면 실패할 수 있으므로, 이를 그대로 전문가 판정 정확도라고 부르지 않는다. 시험 후 오류 검토는 별도로 남기고 고정 채점 기준이나 프롬프트를 수정해 같은 시험을 다시 돌리지 않는다.

동일 문항의 McNemar 수치는 참고용이다. 규칙 내 상관과 다중 비교를 해결한 유의성 검정으로 사용하지 않는다. 핵심 비교는 대응 문항의 차이와 규칙 그룹을 고려한 불확실성이다.

## 실행 순서

기존 학습 체크포인트가 있어야 한다. `outputs/raft-step-opsd-smoke-v1/sft_adapter`와 `final_adapter`를 사용한다.

1. [공식 llama.cpp b11960](https://github.com/ggml-org/llama.cpp/tree/b11960)의 `convert_lora_to_gguf.py`, `conversion/`, `gguf-py/`를 같은 폴더 구조로 준비한다. 이 실행의 tarball SHA-256은 `reports/holdout_environment.json`에 기록했다. 해당 태그의 GGUF 패키지도 설치한다: `.venv/bin/python -m pip install runtime/llama-conversion-b11960/llama.cpp-b11960/gguf-py`.
2. 아래 명령으로 어댑터를 변환한다. 학습에 사용한 HF 기본 모델 설정을 지정해야 한다.

```bash
.venv/bin/python runtime/llama-conversion-b11960/llama.cpp-b11960/convert_lora_to_gguf.py \
  --base models/Qwen3-4B-HF --outtype f32 \
  --outfile outputs/sft-adapter-f32.gguf \
  outputs/raft-step-opsd-smoke-v1/sft_adapter
.venv/bin/python runtime/llama-conversion-b11960/llama.cpp-b11960/convert_lora_to_gguf.py \
  --base models/Qwen3-4B-HF --outtype f32 \
  --outfile outputs/opsd-adapter-f32.gguf \
  outputs/raft-step-opsd-smoke-v1/final_adapter
```

3. 기존 서버를 정상 종료한 뒤 어댑터를 로드한 평가 서버를 시작한다. 실행기는 매 요청의 scale을 명시적으로 설정한다.

```bash
./serve.sh --lora outputs/sft-adapter-f32.gguf,outputs/opsd-adapter-f32.gguf \
  --lora-init-without-apply --no-cache-prompt
```

4. 한 번만 시험 입력을 고정하고 실행한다. 이미 있는 출력 폴더에는 덮어쓰지 않는다.

```bash
python scripts/build_holdout.py --output outputs/evaluation-v1
python scripts/run_holdout.py --dataset outputs/evaluation-v1 --output outputs/evaluation-run-v1
```

`protocol.json`은 자료와 모델을 고정한다. `run_lock.json`과 `requests.jsonl`은 첫 시험 요청 전에 전체 요청 순서·내용과 코드 해시를 기록한다. 생성 중 진행 로그에는 정오답을 출력하지 않고, 모든 요청이 끝난 후 채점한다. 요청 오류도 삭제하거나 성공할 때까지 재시도하지 않는다.

5. 실행 후 기존 서버를 복구한다. 실행기는 완료/실패 시 두 어댑터를 비활성화한다. 운영용 기본 서버는 기존 `./serve.sh`로 다시 시작할 수 있다.

이 v1은 결과가 공개된 이후 개발 분석용으로 사용할 수 있지만, 그 결과로 수정한 모델의 새로운 최종 시험셋으로 재사용하면 안 된다. 다음 개선의 확인에는 새 규칙 그룹과 독립 검수가 필요하다.

## LLMLingua-2 보충 비교

세 모델의 답변을 열어보기 전에 별도 프로토콜을 고정했다. 같은 서술형 8문항의 검색 결과에 기존 조항 보호 방식과 유지율 0.5를 적용하고, 최종 OPSD 어댑터의 입력만 비교한다. 압축 모델은 기존에 설치한 공식 다국어 LLMLingua-2 가중치이며 새로 학습하지 않는다.

```bash
python scripts/evaluate_holdout_compression.py prepare
# 기본 108회 생성 완료 후 실행한다. CPU 경합을 피하기 위해 동시에 실행하지 않는다.
.venv/bin/python scripts/evaluate_holdout_compression.py run
```

전체 요청 JSON이 압축 전과 완전히 같으면 이전 OPSD 응답을 재사용한다. 이것은 추가 독립 추론으로 세지 않는다. 달라진 요청은 같은 모델·생성 설정으로 한 번 생성하고 동일한 고정 기준으로 채점한다. 압축 전처리 시간·토큰 수·입력 변경 여부를 남기며, 재사용 결과나 순차 실행만으로 전체 응답의 속도 개선률을 주장하지 않는다.

이번 평가의 질문 수는 기존 소규모 학습 예시 5개와 60:40 비율을 맞춰 고른 것이 아니다. 이미 학습을 마친 체크포인트에 대해 새로운 규칙 그룹을 분리한 사후 일반화 시험이다. 따라서 처음 제안한 60:40 데이터 분할 실험을 완료했다고 표현하지 않는다.

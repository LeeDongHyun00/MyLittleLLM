# 근거 충분성·규칙 추론·주장 검증

`--pipeline grounded`는 기존 Qwen3-4B Q4_K_M 서버에 붙이는 실험 경로다. 모델 가중치를 새로 학습하지 않았다. 정책 원문에 연결한 규칙 AST, Z3 판정, 제한된 사전 보류 검사, 생성 후 주장 검증을 추가했다. 기존 CLI 기본값은 유지한다.

현재는 [한국어 조건 수집과 입시 규칙 확대](guided_grounding.md)로 규칙 9개와 `--guided-rule` 문답 입력을 제공한다. 아래 18개 시나리오 결과는 최초 6개 규칙으로 수행한 고정 평가 기록이다.

[실제 비교 결과](../reports/grounding_report.md)는 기본 모델 13/18, 규칙·보류 18/18, 전체 결합 15/18이다. 자동 기준이며 독립 전문가 판정 정확도가 아니다. 결과에 따라 `grounded`는 규칙·보류 경로를 기본으로 제공하고 신경망 검증은 `--claim-check`를 명시할 때만 사용한다.

## 실제 적용 범위

| 연구 | 구현한 부분 | 구현하지 않은 부분 |
|---|---|---|
| [Sufficient Context, ICLR 2025](https://arxiv.org/abs/2411.06037) | 생성 전에 근거 없음·개인 처리 기록 필요·규칙 입력 부족을 구분해 보류 | 논문의 일반 충분성 판정기, 학습된 범용 질문 분해기 |
| [NL2Logic, Findings EACL 2026](https://aclanthology.org/2026.findings-eacl.317/)·[Logic-LM, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.248/) | 원문 검토 → 명시적 AST → 제한된 연산 컴파일 → Z3 추론 → 정해진 문장 출력 | LLM을 이용한 자동 자연어→논리식 변환, 논문의 전체 파이프라인 재현 |
| [MiniCheck, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.499/) | 공식 770M 모델을 실제 설치하고 동일한 입력 형식·분류 로짓으로 한국어/영어 개발 검증 | 한국어에서 검증하지 않은 모델을 자동 정답 승인기로 사용 |
| 다국어 NLI 대안 | mDeBERTa의 원문/주장 지지·모순 점수로 자유 서술 답변을 보류시키는 보조 검사 | 높은 점수가 진실을 보장한다는 주장, 한국어 신뢰도 보정 성공 |

Trust-Align·DPO, 부정 대비 학습, RINSE의 내부 표현 판정, conformal 보정은 이번 구현에 포함하지 않았다.

## 실행

```bash
.venv/bin/python -m pip install -r requirements-grounding.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/install_grounding_models.py
./serve.sh
```

기존 원문·색인·Qwen 설치가 필요하다. 검증 모델은 고정된 Hugging Face 리비전에서 받으며 원격 Python 코드를 실행하지 않는다. 모델 파일은 Git에서 제외한다.

원거리 부분 요건 판정:

```bash
.venv/bin/python chat.py --pipeline grounded --task housing --year 2026 --semester 1 \
  --institution 한국장학재단 --facts data/examples/grounded_county.json \
  --question '광역교통권 밖 같은 군 지역이면 원거리 요건을 충족하나요?'
```

조건부 금액 계산:

```bash
.venv/bin/python chat.py --pipeline grounded --task housing --year 2026 --semester 1 \
  --institution 한국장학재단 --facts data/examples/grounded_payment.json \
  --question '입력한 4월 인정 지출과 승인 조건에 따른 지원액을 계산해 주세요.'
```

자유 질문은 `--facts` 없이 실행한다. 규칙 판정·계산은 Qwen을 호출하지 않고 조건부 결론을 정해진 문장으로 출력한다. 자유 서술은 BM25 상위 2개, minimal 프롬프트, JSON 스키마, 384 출력 토큰으로 생성한다. `--claim-check`를 추가할 때만 신경망 검증 모델을 로드해 인용 근거와 대조한다. 이 실험 옵션은 정답을 거절할 수 있으며 이번 시험에서는 성능이 낮아졌다. 신경망 검색 모델은 동시에 로드하지 않는다. 일반 `grounded`에는 신경망 검증이나 사후 지급 표현 검사가 없고, 사전 보류 검사와 규칙·계산 분기에서만 해당 보호가 적용된다.

## 구조화 입력의 의미

자연어 질문에서 조건을 자동으로 추출하지 않는다. `--facts`에 명시한 값은 사용자가 제공한 시나리오 조건이며 외부 기록을 조회해 확인한 사실이 아니다. 실제 상담 UI에는 이 필드를 사용자에게 이해되는 질문으로 수집하는 단계가 추가로 필요하다. 질문과 구조화 입력 간 모든 의미 충돌을 탐지하는 기능은 없다.

| rule_id | 판정 범위·주요 필드 |
|---|---|
| housing_academic | 성적·학점 부분 요건. `ordinary_semester_system=true`는 학기제 확인, `first_semester_exempt`는 신·편입/재입학생의 첫 학기 면제, `disabled`, `care_leaver`, `graduation_or_excess`는 해당 예외. `credits`는 인정 취득학점, `required_credits`는 학교 규정으로 확인한 최소 학점, `score`는 백분위 정수 점수다. 학년제·소수 학점/점수는 현재 범위 밖이다. |
| housing_expense_combination | 보증금 월환산액과 주택임차차입금 이자의 동시 청구 제한. `deposit_conversion`, `housing_loan_interest`, `rent_loan_interest`로 구분한다. 월세대출 이자를 주택임차차입금 이자와 혼동하면 안 된다. |
| housing_duplicate_support | `same_year_month`, `other_program_is_conflicting`, `overlap_days`로 동일 연월의 제한 대상 사업과 자격 중복을 판정한다. 타 사업이 제한 대상인지 자동 분류하지 않는다. |
| housing_benefit_count | `academic_years`(2–6), 규정에 맞춰 계산한 현 학교 실적 `school_used`, 모든 학교의 주거장학금 누적 실적 `personal_used`로 다음 1회 수혜 한도를 판정한다. 전과·재입학과 신·편입학에 따른 실적 산정은 사용자가 먼저 확인해야 한다. |
| housing_county_distance | `outside_metro_county=true`일 때만 사용한다. `same_county=true`는 부모 중 대학 소재 군 범위에 거주하는 사람이 있음을 뜻한다. `exception_approved`는 대학·재단의 실제 예외 인정 여부다. 시·광역교통권의 다른 지역 경계는 다루지 않는다. |
| admissions_ged_certificate | 부산대 2026 수시, `ged_offline=true`인 검정고시 온라인 제공 미동의·불가자만 적용한다. `essay_track`은 논술 여부, `score_certificate=true`는 성적증명서, false는 합격증명서다. 다른 증명서 종류에는 사용할 수 없다. |

주거 규칙에는 `year=2026`, `semester=1`, `task=housing`, `institution=한국장학재단`도 넣는다. 입시 규칙에는 `year=2026`, `task=admissions`, `institution=부산대학교`를 넣는다. 원문 해시와 핵심 구절이 달라지면 규칙을 적용하지 않는다. 원문 검수는 AI 도우미가 했으며 독립 전문가 승인은 받지 않았다.

필드가 없으면 false로 간주하지 않는다. Z3가 조건의 참·거짓 중 하나를 증명하지 못하면 `unknown`으로 보류한다. 다만 장애인 면제처럼 다른 변수를 몰라도 결론이 증명되는 경우에는 판정할 수 있다. 최종 입학·장학금 선발이나 실제 송금 여부를 판정하는 규칙은 없다.

학교별 수혜 횟수가 모든 학교의 개인별 누적 수혜 횟수보다 큰 입력은 모순으로 보류한다. 이 불변 조건과 설명 문구 보완은 평가 완료 후 반영했다.

## 검증 모델의 실제 개발 결과

긍정·부정/조건 변형 한국어 8쌍(16개 주장)과 영어 1쌍을 실행했다. 자동 승인 채택 조건을 한국어 정답 6/8 이상 승인, 오답 승인 0/8로 먼저 정했다. 두 모델 모두 조건을 통과하지 못했다.

- MiniCheck-Flan-T5-Large: 0.5 임계값에서 한국어 정답 승인 0/8, 오답 승인 0/8. 한국어의 많은 토큰이 unknown으로 변환되어 원문과 반전 문장을 구별하기 어려웠다. 한국어 답변의 승인·차단 기준으로 채택하지 않았다.
- [mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli): 0.5에서 정답 승인 8/8, 오답 승인 2/8. 잘못된 보관 기간 시작점과 실제 지급 완료 주장에 높은 지지 점수를 줬다. 0.95까지 높여도 이 두 오류가 남았다.

한국어 개발 입력의 unknown 토큰은 MiniCheck 222/542, 다국어 NLI 0/674였다. 영어 확인용 1쌍은 두 모델 모두 구분했다. 이는 전체 언어 성능 평가가 아니라 이 프로젝트에서 모델을 잘못 연결하지 않기 위한 작은 개발 점검이다.

따라서 다국어 NLI는 보조적인 거절 필터다. 지지 점수 0.5 미만이거나 모순 점수 0.8 이상이면 보류하지만, 통과했다고 정답을 보증하지 않는다. 실제 지급 완료 표현에는 별도 코드 검사도 적용한다. 검수된 규칙·계산의 결과를 신경망 점수로 뒤집지 않는다.

인용된 문서 구간을 문장별 주장과 비교한다. 512토큰을 넘는 근거는 잘라서 예외를 버리지 않고 검증 불가로 취급한다. 긴 인용 때문에 맞는 답도 보류될 수 있다. 문장 안의 복수 주장·조건 범위를 완전하게 분해하는 RefChecker는 구현하지 않았다.

## 평가

```bash
.venv/bin/python scripts/prepare_grounding_evaluation.py
.venv/bin/python scripts/evaluate_grounding.py --stage prepare
.venv/bin/python scripts/evaluate_grounding.py --stage generate
.venv/bin/python scripts/evaluate_grounding.py --stage evaluate
```

이미 생성한 파일은 덮어쓰지 않는다. 생성 중단은 `--stage generate --resume`으로 이어간다. 공식 원문과 코드, 요청, 모델, 정답을 고정한 뒤 생성한다. 생성 결과를 확인한 후 프롬프트·규칙·정답 기준을 바꾸지 않는다.

고정 평가의 코드 커밋은 `829f712af32875ada9e825de7ab0a718a624710d`다. 완료 후 입력 모순 검사·조건부 설명·CLI의 검증 선택 옵션을 수정했으므로 현재 코드에서 과거 실행을 재개하면 의도대로 잠금 해시 검사가 실패한다. 고정 결과 검증·재현은 해당 커밋으로 수행해야 한다. [후속 수정 감사](../reports/grounding_post_eval_check.json)에서 18개 요청과 판정 상태·결론·인용이 동일함을 확인했다. 설명 문구 4개는 변경했지만 과거 응답이나 점수를 수정하거나 새로 생성하지 않았다.

18개 새 시나리오는 검수한 규칙의 새로운 적용 사례와 자유 질문이다. 새 규칙·새 대학으로 분리한 시험이 아니다. 정답과 구조화 조건은 AI가 작성하여 독립 검수하지 않았다. 규칙 선택은 구조화 입력의 `rule_id`로 제공한다. 자연어 입력만으로 얻는 운영 정확도로 해석할 수 없다.

비교 조건은 동일 근거·입력을 받는 Qwen, Qwen+검증 필터, 규칙·보류 검사, 전체 결합이다. 동일한 Qwen 요청을 한 번 생성해 후처리 조건에 재사용한다. 규칙 판정에는 생성 호출이 없다. 정규식 점수와 전체 답변의 별도 의미 검토를 함께 공개하며 답변률·불필요 보류·잘못된 확정 답변을 분리한다.

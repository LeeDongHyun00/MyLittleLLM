# L01 부분 규칙·문답 경계와 false 설명

2026-11-04, S3 / `s3/l01-rule-boundaries`.
시작 커밋: `a921e39ba53d6b579156262b13e898a9f63669bd`.

9개 부분 규칙의 결정식은 유지하고, `false` 결론의 이유를 확인할 수 있도록
선택적 설명을 추가했다. 신규 16개 테스트와 기존 규칙·문답·runtime 36개를 합친
52개 테스트가 통과했다. 이 중 9규칙 × true/false/unknown의 27개 사례는
[개발 회귀 스냅샷](l01_rule_boundaries.json)에 실제 입력과 결과를 남겼다.
이는 AI 작성 공개 개발 사례의 예상 동작 검증이며 새 평가 정확도나 H1 승인 결과가 아니다.

## 변경과 호환성

[규칙 설정](../config/grounding_rules.json)에 `false_explanation` 목록을 선택적으로 추가했다.
각 항목은 `conjunct_path`와 `text`를 갖는다. 빈 경로 `[]`는 전체 expression,
`[0]`은 최상위 AND의 첫 조건을 뜻한다. 경로는 AND만 통과할 수 있으므로
OR 안의 한 대안이 거짓이라는 이유로 전체 요건이 실패했다고 설명할 수 없다.
조건 자체는 기존 AST를 참조하며 설명에 별도 판정식을 복제하지 않는다.

[PolicyRules](../mylittlellm/policy_rules.py)는 전체 결론이 `false`일 때에만 설명을 준비한다.
입력과 정수 도메인·제약의 일관성을 확인하고, 각 대상 조건을 추가했을 때 Z3가
`unsat`을 입증한 문장만 기존 `explanation` 앞에 붙인다. 해당 하위 조건이
미확인 또는 solver `unknown`이면 그 문장은 붙이지 않는다. 모순 입력도 설명을 만들지 않는다.
한 필요 조건의 확정적 실패로 전체 결론이 정해져도 다른 미확인 조건의 값은 채우지 않는다.

예를 들어 학력 조건만 명시적으로 아니라고 입력하고 법정 지위를 생략한 경우,
학력 미달 이유만 보여 준다. 수급권자·차상위·한부모 법정 지원대상이 아니라고 추정하지 않는다.
검정고시 규칙의 `false`는 논술전형 성적증명서 제출 **면제**이므로 불합격이나 요건 미충족으로 쓰지 않는다.

이 변경은 S0가 승인한 선택적 rule-config 확장이다. 기존 소비자는 추가 항목을 읽지 않아도
기존 판정을 유지한다. 새 소비자도 항목이 없으면 기존 설명을 그대로 사용한다.
외부 JSON과 proof의 필드 집합, 규칙 ID·AST·scope·constraints·applicability·source SHA·anchors,
review_status는 유지했다. 불충족 설명 문자열이 달라지므로 문자열 전체 일치에 의존한 소비자는
새 동작을 확인해야 한다. 설정의 의미와 한국어 설명의 일치는 아직 독립 검수되지 않았다.
S0가 공통 INTERFACES/ADR 및 계약 버전 채택을 담당하며 이 작업의 계약 스냅샷은 1.0.0이다.

## 조항 연결 경계 사례

모든 기본 행은 해당 규칙의 scope와 applicability를 명시적으로 확인한 가상 입력이다.
추가 테스트에서는 applicability 누락과 명시적 `false`를 각각 확인해
`missing_applicability` / `outside_rule_scope`와 `unknown`으로 보류함을 검증했다.

| 규칙 | true / false / unknown의 대표 경계 | 기존 연결 조항 |
|---|---|---|
| `housing_academic` | 첫 학기 면제 / 면제 조건 없음·69점 / 졸업학년 학점 면제·성적 미확인 | `kosaf-2026-housing-plan-faq:p9:block0`, `p9:block1` |
| `housing_expense_combination` | 보증금+월세대출 이자 / 보증금+주택임차차입금 이자 / 주택임차차입금 이자 여부 미확인 | 같은 자료 `p10:block2` |
| `housing_duplicate_support` | 자격 중복 0일 / 제한 대상·동일 연월·1일 / 중복 일수 미확인 | 같은 자료 `p10:block0` |
| `housing_benefit_count` | 6년제·학교/개인 11회 / 4년제·학교 7회·개인 8회 / 개인 횟수 미확인 | 같은 자료 `p11:block0`, `p11:block1` |
| `housing_county_distance` | 같은 군 아님 / 같은 군·예외 인정 없음 / 예외 인정 미확인 | 같은 자료 `p8:table0:row32`, `p5:block0` |
| `admissions_ged_certificate` | 논술·합격증명서 제출 / 논술·성적증명서 면제 / 증명서 종류 미확인 | `pusan-2026-susi:p24:block1` |
| `admissions_susi_application_limit` | 기존 산입 5회 / 6회 / 산입 횟수 미확인 | 같은 자료 `p27:block0` |
| `admissions_low_income_status` | 학력+법정 대상 확인 / 학력 조건 아니오·법정 지위 미확인 / 한부모 법정 지원대상 미확인 | 같은 자료 `p54:block1` |
| `admissions_vocational_subject` | 인정 학교·관련 교과 30단위 / 29단위 / 관련 교과 수치 미확인 | 같은 자료 `p54:block1` |

추가 회귀는 마이스터고·체험 위주 학교 제외를 동일계열 이수나 300단위 입력이 상쇄하지 않는지,
학교 실적 8회·전체 실적 2회 모순이 `unknown`으로 남는지, 한부모 가구라는 질문 문장을
법정 자격으로 자동 변환하지 않는지 확인한다. 모든 bool/int 필드의 null·잘못된 타입·범위 밖 값을
거부하며 문답은 5.9를 반올림하지 않고 다시 묻는다. 명시적 0은 0으로 보존하고 모름·중단은
필드를 생략한다. 기존 guided_facts 구현 자체는 수정할 필요가 없었다.

각 규칙의 true 결과에도 최종 합격 확정 요청을 덧붙이면 `cannot_determine`으로 보류한다.
주거 성적 부분 판정으로 실제 입금 여부를 답하지 않는 기존 개인 기록 게이트도 통과했다.

## 원문·검수 한계

테스트 색인은 기존 source ID·SHA와 anchors를 조합한 **합성 레코드**다.
원문 PDF를 재구축하거나 그 의미를 검수한 결과가 아니다. 원문 해시/앵커 불일치 시 거부하는
코드 경로를 테스트할 뿐, 합성 레코드의 해시 문자열을 실제 PDF의 무결성 증거로 사용하지 않는다.

- 기존 주거 PDF SHA: `8e4c1f6bee6071db509aba3032ba1afb1c907d24a82f08d1465350a8ecbb4c9c`.
- 기존 부산대 PDF SHA: `a28a53b010a350b4ce6df667f832309ccd8c4913d09edb9236c1bd64a757b5d3`.
- D01 동서대 미러 SHA `22af5aaa41f0d32377f4c5adbe682de502f1e245712cf44432e448e312b83aa2`는
  다른 원문이다. 이 작업에서 기존 source hash를 미러로 바꾸지 않았다.
- 원문 수집·정오표 대조·실제 청크 재연결, H1 독립 검수, 새 잠금 평가, 모델 실행·색인·학습은 수행하지 않았다.

Z3는 선택된 AST 조건의 거짓을 입증한다. 한국어 설명의 사실성이나 원문의 정책 해석까지
검증하지 않는다. H1은 최종 원문과 설명 문장을 따로 대조해야 하며, 향후 AST를 바꾸면
conjunct_path와 수치가 포함된 설명도 함께 재검토해야 한다.

설명 준비에는 false 결과마다 solver 실행이 추가된다. 현재 최대 4개 설명이 있는 특성화고
규칙은 일관성 확인 1회와 조건 확인 4회, 최대 5회의 추가 `check()`를 사용한다.
각 check의 기존 timeout은 1,000ms다. 전체 요청 deadline은 새로 구현하지 않았다.
테스트 실행 시간은 로컬 회귀 확인용이며 모델·실서비스 지연 측정이나 성능 우위 증거가 아니다.
E02에서는 공통 L 버전을 모든 arm에 고정하고 이 비용도 전체 지연에 포함해야 한다.

## 검증과 다음 작업

전용 venv `/tmp/mllm-s3-l01-venv`에 `z3-solver==4.15.4.0`만 설치했다.
기본 sandbox 네트워크 권한 없는 첫 설치는 실패했고, 명시적 추가 네트워크 권한을 사용한 설치는 성공했다.
다른 세션 Python 환경은 변경하지 않았다. pytest·모델 의존성 추가 설치는 하지 않았다.

```bash
PYTHONPATH=tests /tmp/mllm-s3-l01-venv/bin/python -m unittest test_grounding test_guided_grounding test_l01_boundaries test_runtime_logic -v
python scripts/check_project_contract.py --handoff docs/handoffs/2026-11-04-s3-l01-rule-boundaries.json
git diff --check
```

52개 테스트 통과, 기본 계약 검사 통과, diff 공백 검사 통과.
설정에서 `false_explanation`만 제거한 값과 시작 커밋의 설정이 완전히 일치하는 것도 확인했다.
전체 모델 평가나 52개 이외의 무관한 테스트를 이번 작업에서 통과했다고 주장하지 않는다.

S0는 선택적 설정 필드를 공통 계약에 기록하고 후보 커밋을 통합한다.
S5/E02는 동일 L 버전 동결, H1·보관자 지정, 원문 확인과 새 독립 데이터가 완료될 때까지
최종 평가를 열지 않는다. 이 보고서의 27개 개발 사례는 독립 시험 후보로 재사용하지 않는다.

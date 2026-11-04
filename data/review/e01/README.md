# E01 — H1 독립 검수 작업 안내

이 패키지는 **미작성 양식**이다. H1은 아직 미확보이며 승인된 gold는 0건이다.
AI나 통합 담당자가 사람의 이름·날짜·답안을 대신 기입하지 않는다.
[E01 프로토콜](../../../docs/evaluation/e01_protocol.md)과
[후보 manifest](../../evaluation/e01/candidate_manifest.json)를 기준으로 사용한다.
과거 [v2 검수 양식](../grounding_v2_blind_review.csv)은 그대로 보존하며 공개 회귀 자료로만 검수한다.

## 보관자가 먼저 할 일

1. 실제 검수자 두 명과 조정 담당을 지정한다. 해당 범위의 원문을 해석할 수 있는 경험,
   데이터/규칙/모델 개발 관여 여부와 이해 충돌을 private roster에 적는다.
   두 H1은 평가 대상 구현·AI gold 작성에 참여하지 않아야 한다. 독립성이 없으면 별도 내부 검토로 표시한다.
2. 신원·연락처·역할·담당 범위의 private 대응표를 보관한다. Git에는 실제 개인정보를 넣지 않는다.
   공개 보고는 동의된 pseudonymous reviewer ID와 검수 여부만 제공한다.
3. 접근 제한된 별도 장소에 검수자별 사본을 만든다. 공유 Git/공유 workspace는 잠금 장소가 아니다.
   파일 digest, 수신자, UTC 시각, 목적을 [접근 장부](access_log_template.csv)에 적는다.
4. 질문과 명시된 사용자 조건, 적용 대학·학년도·학기·전형·as_of_date,
   공식 원문 전체·수정공고 목록을 배포한다. 원문 SHA/URL/수집일/버전은 같이 주되
   개발자가 고른 정답 쪽·gold 청크·제안 정답·AI 해설·모델 출력은 주지 않는다.
   다운로드 실패 또는 적용 수정본 미확인은 답을 추정하지 말고 자료 확인 대기로 둔다.

## 단계 A — 모델 출력 없이 gold 작성

[gold_review_template.csv](gold_review_template.csv)를 각 검수자에게 독립 복사한다.
헤더만 있는 파일이다. 후보 ID도 실제 문항 확정 전까지 생성하지 않는다.

- `candidate_id`, 질문, 범위/명시 조건, 원문 목록은 보관자가 채운다.
  `scope_json`은 year/task/institution/semester와 admission_track/as_of_date를 담는다.
  `user_conditions_json`은 사용자 제공 조건이지 확인된 실제 학생 기록이 아니다.
  모르는 조건은 생략하고 문항에서 미확인이라고 표시한다.
- 검수자는 먼저 `answerability`(answerable/missing_conditions/no_authoritative_evidence/
  outside_scope/conflicting_sources)와 `expected_action`(answer/clarify/abstain)을 판단한다.
  이후 자신의 결론 `reviewer_answer`, 필요한 사실/질문, 금지 주장, 부분 판정을 작성한다.
  부분 판정이 적용되지 않으면 `expected_partial_decision`은 `not_applicable`로 남긴다.
- `reviewer_evidence_json`에는 source_id, URL, SHA-256, PDF 파일 쪽(1부터), 인쇄 쪽(선택),
  조항/표/각주 위치, 적용 시기와 확인한 핵심 구절의 최소 발췌를 적는다.
  허용 대체 근거가 있으면 필수 조항 묶음들을 `acceptable_evidence_sets_json`에 적는다.
  근거 없는 질문은 빈 근거 배열과 조사한 원문 범위·보류 이유를 남긴다.
- 표의 열 제목·각주·예외·수정본을 대조한다. 모순을 임의로 해소하지 않는다.
  '조건 충족'은 부분 요건일 뿐 최종 합격/선발/입금 완료가 아니다.
  계산은 원문에 근거해 사람이 별도 산출하고 금액·기간·단위·반올림 규칙을 남긴다.
  프로그램 출력을 정답으로 복사하지 않는다.
- `model_outputs_seen_before_submission=false`, `ai_answers_seen_before_submission=false`,
  `peer_answers_seen_before_submission=false`는 실제 미노출일 때만 사람이 기입한다.
  노출됐다면 true와 내용을 남겨 독립 검수 부적격으로 처리한다. 빈칸은 false가 아니다.
- `reviewer_id`, `reviewed_at_utc`, `submission_sha256`은 실제 제출을 기록한다.
  digest는 순환 참조를 피하기 위해 **이 digest 열을 비운 제출 파일 바이트**의 SHA-256이다.
  원본 제출 파일은 보존하고 digest가 채워진 접수 장부/사본을 따로 만든다.
  최초 상태는 `submitted` 또는 `needs_source_check`이며 검수자 한 명이 `approved`로 승격하지 않는다.

두 답안이 제출·봉인되기 전에는 상대 답이나 모델 출력을 보여주지 않는다.
모델 답이 없는 상태에서도 두 사람의 action/결론/근거/필수 조건/범위가 다르면 불일치다.

## 단계 B — 불일치 조정과 승인

[adjudication_template.csv](adjudication_template.csv)에 검수자 두 명의 제출 경로/digest와
불일치 항목을 연결한다. 조정은 제3의 적격 H1을 우선하며, 미확보 시 두 원검수자의
원문 기반 합의와 양쪽 실제 sign-off를 기록한다. 합의 실패는 `excluded` 또는 `needs_revision`이다.

- 합의가 있어도 근거의 수정본·적용 시기·표/각주·질문 모호함을 확인한다.
  다수결/정규식 일치/LLM judge를 승인 근거로 쓰지 않는다.
- 문항·조건 수정은 새 revision으로 만들고 두 검수자가 모델 없이 다시 검수한다.
  기존 답안을 덮어쓰지 않는다. 한 사람 제출만으로 승인하지 않는다.
- 조정 장부는 최종 action, answer, required_facts, forbidden_claims, 필요한 질문,
  부분 판정, 허용 근거/오차, 변경/제외 사유와 실제 조정 일자를 담는다.
  두 검수자의 sign-off가 모두 있고 자료·모호함 문제가 해결됐을 때만 보관자가
  `review_status=approved`를 부여한다. 실제 서명은 private 기록으로 증빙한다.
- 승인 gold와 노출/누출 그룹 승인은 별개다. 승인된 회귀 문항은 계속 회귀 문항이다.
  public manifest에는 합계/상태/안전한 digest만 갱신하며 시험별 답/그룹별 근거는 공개하지 않는다.

그룹 감사는 [leakage_audit_template.csv](leakage_audit_template.csv)에 연결 이유,
양쪽 규칙/연도/복사본/템플릿, 기존 노출 여부, 결정자·날짜를 남긴다.
연결 없음도 어떤 기존 자료를 점검했는지 기록한다. 기존 누출 그룹에 연결되면 회귀로 분류한다.
새 대학 일반화라고 바꾸려면 대학 전체와 전국 공통 규정의 묶음을 다루는 별도 프로토콜이 필요하다.

## 단계 C — 시험 동결 후 응답 채점

gold 승인·분할·후보 설정 동결 후에만 모델을 실행한다.
[response_review_template.csv](response_review_template.csv)는 단계 A와 **별도** 자료다.
보관자가 arm·모델명을 가린 response ID와 원시 응답/실제 근거를 제공하고 매핑은 따로 보관한다.
두 검수자는 자동 점수·서로의 점수를 보기 전에 전 문항을 독립 채점한다.
JSON 오류/잘림/보류도 제거하지 않는다. 형식상 블라인드여도 문체로 arm이 추측될 수 있는 한계를 기록한다.

CSV의 boolean 채점 칸은 true/false로 기입하고 해당 없음은 `not_applicable`, 미확인은
`unknown`으로 구분한다. 오류 출력의 `strict_pass`는 false다. 제출 digest는 단계 A와 같이
digest 열을 비운 원본에 대해 계산하고 원본과 접수 사본을 모두 보존한다.

실질 행동, 필수 사실, 인용 지지, 계산, 부분 결론, 불필요 보류,
잘못된 승인/배제/미확인 단정, 오류 태그, strict_pass와 사유를 기록한다.
strict는 [프로토콜](../../../docs/evaluation/e01_protocol.md)의 행동별 기준을 모두 만족할 때만 true다.
두 사람의 최초 일치 건수/분모와 불일치 목록을 보존하고 조정 점수는 새 사본에 쓴다.
새 gold 오류를 발견하면 최초 점수·gold를 보존하고 정정 분석을 별도로 만든다.
출력을 보고 gold를 바꾼 결과를 사전 고정된 독립 시험 성능으로 발표하지 않는다.

## 준비 완료와 실행 대기

양식의 0행은 의도된 상태다. 빈 이름/답/날짜를 임의로 채우지 않는다.
H1·보관자 미지정, 원문/누출 감사 미완료, 후보 0건이므로 E02는 대기다.
이 안내나 체크 통과는 실제 검수·신원·의미 정답·접근 통제를 인증하지 않는다.

보관자가 별도 비공개 checkout에서 모든 승인 증거를 확인한 뒤에만 다음 형식으로 분할한다.
`/PRIVATE_ROOT`는 보관자가 확인한 실제 접근 제한 경로로 바꾼다. 현재 실행을 지시하는 명령이 아니다.

```bash
python scripts/split_dataset.py /PRIVATE_ROOT/e01/approved-selected-25-groups.jsonl \
  --output /PRIVATE_ROOT/e01/splits-v1 --group-key leakage_group --seed 20261104
```

출력 폴더가 이미 있으면 새 경로를 사용한다. train/dev 공개 전에 test를 별도 권한으로 분리하고
분할 보고/각 파일 해시/그룹 교차 여부를 확인한다. 스크립트 자체에는 이 권한 관리 기능이 없다.
기존 splitter는 answer/evidence의 존재도 요구한다. `abstain`에서 지지 근거가 없으면 조사한
원문 범위를 evidence에 `not_supporting_answer=true`로 명시하고 실제 허용 지지 근거 묶음은 빈 배열로 둔다.
이 기록을 검색 정답 조항이나 인용 지지 성공으로 집계하지 않도록 E02 adapter에서 검증해야 한다.

# 현재 인터페이스와 변경 경계

계약 1.1.0. 기존 구현 기준 `d0ac404c8cda106e703bd04a691aaf8e10abf056`에 L01의 선택적 설명 확장을 통합했다.
아래는 **현재 Python 구현을 설명하는 계약**이다. 모든 항목을 별도 JSON Schema로 강제하는 구현은 없다.
향후 필드 변경은 S0 결정 기록과 생산자/소비자 테스트를 함께 갱신한다.

## 원문 → 검색 근거

[원문 목록](../../data/sources.jsonl)은 `source_id`, `title`, `organization`, `academic_year`,
`semester`, `source_type`, `format`, `url`, `landing_url`, `checked_on`, `label_status`,
`version_status`, `license_note` 등을 기록한다. 원문 목록의 연도 필드는 `academic_year`이고,
질의/규칙 범위의 연도 필드는 `year`다. 이를 같은 필드명이라고 가정하지 않는다.

현재 검색/판정에 쓰는 근거는 `chunk_id`, `source_id`, `text`, `pdf_page`, `sha256`, `url`과
적용 범위 메타데이터를 포함한다. `pdf_page`는 PDF 파일 기준 쪽이며 인쇄된 쪽번호와 다를 수 있다.
소스 해시는 같은 문서 조각에서 반복될 수 있다. `chunk_id`는 생성된 색인 내부 식별자다.
S1은 쪽/표/각주를 보존하고 S2는 최종 모델에 넘긴 실제 근거와 순서를 기록한다.

표의 셀 좌표·각주 링크를 독립 필드로 보존하는 확장은 **D01의 제안**이며 아직 전역 스키마가 아니다.
원문 전체·검색 문맥 전체는 재배포하지 않고 URL/해시/검증에 필요한 소량 발췌를 남긴다.

## 질의 → 규칙/모델

현재 질의 `row`는 `question`, `year`, `task`, `institution`, 선택적 `semester`를 받는다.
grounded의 질문 길이는 1–2,000자다. `task` 예시는 `admissions`, `housing`이다.
규칙 조건은 `facts` 객체에 `rule_id`, 해당 규칙의 범위, 정의된 조건 필드를 전달한다.

```json
{
  "rule_id": "admissions_susi_application_limit",
  "year": 2026,
  "task": "admissions",
  "institution": "부산대학교",
  "counted_application": true,
  "counted_applications": 5
}
```

JSON에서 **모르는 규칙 필드는 생략**한다. 명시적 `false`는 거짓 확인, `0`은 실제 0이며
미확인 대용이 아니다. `null`도 bool/int 조건 값으로 유효하지 않다. 문답의 `모름`은 필드를 비워 둔다.
현재 조건은 bool/정수만 지원한다. 소수 성적을 반올림해서 입력하지 않는다.
모르는 필드명·잘못된 타입·범위 밖 수치는 오류 처리한다. 자연어 전체와 조건 사이의 의미 일치는 완전 검증하지 않는다.

## 규칙 → 부분 판정

[규칙 설정](../../config/grounding_rules.json)의 각 규칙은 `id`, `label`, `question_pattern`, `scope`,
`fields`, `expression`, `sources`, `explanation`과 선택적 `constraints`, `applicability`를 가진다.
출처는 `chunk_id`·`sha256`·`anchors`를 확인한다. AST는 허용된 연산만 사용한다.

1.1.0에서 선택적 `false_explanation: [{"conjunct_path": [], "text": "..."}]`을 추가했다.
빈 경로는 전체 expression, 정수 경로는 AND 자식만 참조한다. OR의 한 대안을 실패 요건으로 설명하지 않는다.
전체 판정이 false이고 해당 조건의 거짓이 입증된 경우에만 `proof.explanation` 앞에 문장을 붙인다.
이 필드가 없으면 기존 설명을 사용한다. 기존 규칙 AST/ID/범위/원문 연결과 외부 proof 키는 유지한다.
문자열 전체 일치에 의존하는 소비자는 새 설명을 확인해야 한다. 한국어 문구와 조항의 일치는 H1 검수 대상이다.
false 설명은 최대 5회 solver check를 추가할 수 있으며 전체 요청 deadline은 구현하지 않았다.
[L01 보고서](../../reports/l01_rule_boundaries.md)와 경계 테스트가 구현·호환성 근거다.

`proof.decision`은 문자열 `true` / `false` / `unknown`이다. 의미는 해당 부분 요건의 논리 판정이다.
`proof.reason`은 `symbolic_entailment`, `missing_facts`, `missing_applicability`,
`outside_rule_scope`, `inconsistent_facts`, `solver_unresolved` 등이다.
누락 조건이 결론을 바꿀 수 있으면 추가 질문을 제공하며, solver 시간 초과를 거짓으로 바꾸지 않는다.
원문/스코프 불일치 등의 예외는 외부 응답에서 `invalid_or_unsupported_facts` 보류가 될 수 있다.

## 모델/판정 → 응답

모델 JSON은 `status`, `answer`, `citations`, `follow_up` 네 필드다.
모델의 `status`는 `answer` 또는 `cannot_determine`; `citations`는 제공된 근거의 ID 배열이다.
스키마 제약과 인용 ID 검사는 사실 검증이 아니다.

grounded 최종 응답은 위 필드에 `reason`, `decision`, `questions`, `proof`, `verification`,
`generation`, `sources`, `pipeline`, `llm_called`, `elapsed_seconds` 등을 조건별로 붙인다.
출력이 잘리거나 파싱/검증에 실패하면 시스템이 `status=invalid_response`를 반환한다.
`reason`은 외부 처리 이유이며 `proof.reason`과 동일한 열거형이 아니다.
추가 질문은 현재 `cannot_determine` + `follow_up`/`questions`로 표현한다.
**`clarify`는 평가 설계상의 개념이지 현재 응답 status가 아니다.**

계산용 facts는 규칙 facts와 별도 구조다. [계산 구현](../../mylittlellm/preprocessed_assistant.py)의
`calculation_evidence`와 [비용 함수](../../mylittlellm/costs.py)를 확인한다.
계산 입력을 규칙 필드로 혼합하지 않고 조건부 금액을 실제 지급액으로 표시하지 않는다.

## 평가 → 보고서

평가 표본에는 ID·질의/범위·조건·기대 상태/부분 판정·근거와 채점 기준·검수 상태를 연결한다.
현재 공개 v2 표본은 `review_status=ai_authored_awaiting_independent_review`다.
`reviewed_rule`이라는 내부 응답 이유가 독립 사람 승인이라는 뜻은 아니다.

새 실행은 코드/소스/색인/요청/정답/설정 해시, 실제 모델 호출 수, 재사용 응답 수, 판단 건수를 구분한다.
같은 응답을 여러 판정 경로가 공유해도 모델을 여러 번 호출한 것으로 세지 않는다.
`strict_pass`와 `false_assertions`는 현재 자동 기준의 명칭이며 의미 검토를 대체하지 않는다.
공개 문항으로 선택한 설정은 새로운 독립 사례 평가 전에 고정한다.

문서 자동 검사(`scripts/check_project_contract.py`)는 경로·버전·인수인계 필수 항목·참조 10개·
규칙 ID·동결 파일 해시·대표 점수의 일관성을 확인한다. 런타임 응답/원문 의미를 검사하는 도구는 아니다.

## 세션 → GitHub 이벤트

[세션 목록](../../contracts/sessions.json)의 schema_version은 1이다.
`sessions`는 역할/작업/브랜치/기준 커밋/계약 버전/상태/인수인계/PR을 기록한다.
상태는 커밋 시점의 `running`, `completed`, `partial`, `blocked`이며 완료 여부는 작업 범위에 한정된다.
과거 handoff는 명시된 `historical_contract_versions`로 검사하고 재작성하지 않는다.
[세션 검사](../../scripts/check_session_handoffs.py)는 메타데이터·산출물 경로를 검사한다.
원격 PR 상태·세션 생존·독립 검수·Git 계보 확인은 자동 검사 범위가 아니다.
이벤트 종류, 권한과 외부 웹훅 미연결 상태는 [COLLABORATION](COLLABORATION.md)에 기록한다.

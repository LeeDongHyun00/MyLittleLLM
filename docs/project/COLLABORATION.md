# 세션 협업과 GitHub 이벤트

2026-11-04 사용자가 세션 생성과 Git 웹훅 협업을 명시적으로 요청했다.
이에 따라 현재 대화의 S0/I01과 작업 에이전트 S2/R01·S3/L01을 분리했다.
별도 ChatGPT 채팅방이나 종료 후에도 가동되는 원격 작업 서버를 만든 것은 아니다.
기본 자동 에이전트 생성 금지는 유지하며 이번 분리는 명시적 요청 범위다.

## 작업 위치와 공유 경계

| 역할/작업 | 브랜치 | 이번 환경의 작업 경로 |
|---|---|---|
| S0/I01 | `s0/i01-session-collaboration` | `/workspace/MyLittleLLM` |
| S2/R01 | `s2/r01-retrieval` | `/workspace/worktrees/mllm-s2-r01` |
| S3/L01 | `s3/l01-rule-boundaries` | `/workspace/worktrees/mllm-s3-l01` |

세 작업은 `a921e39ba53d6b579156262b13e898a9f63669bd`에서 분기했다.
선행 결과는 [E01](../handoffs/2026-11-04-s5-e01-evaluation-preparation.json)과
[D01](../handoffs/2026-11-04-s1-d01-source-audit.json)이다.
S2는 별도 개발 검색 후보, S3는 규칙의 부분 판정 설명을 맡는다.
S0가 계약·공통 문서·워크플로·세션 목록·최신 인수인계 포인터를 통합한다.
원문/색인 등 Git 제외 산출물은 자동 공유되지 않는다. 이번에는 D01 별도 배포본을 읽기 전용 입력으로 허용했다.
무거운 생성/학습/대형 색인은 동시에 실행하지 않으며 이번 세션들은 모델을 호출하지 않는다.

현재 커밋에 기록된 상태는 [sessions.json](../../contracts/sessions.json), 협업 이력은
[이슈 #3](https://github.com/LeeDongHyun00/MyLittleLLM/issues/3)을 확인한다.
`running`은 그 커밋 작성 당시의 상태다. 실시간 heartbeat나 다음 대화에서의 자동 재개를 뜻하지 않는다.

## 이벤트가 하는 일

[Session collaboration](../../.github/workflows/session-collaboration.yml)은 역할 브랜치·통합 기준 브랜치의
`push`, 그리고 PR 생성/수정(`pull_request`) 이벤트로 실행된다.
GitHub가 제공하는 기본 토큰에는 `contents: read`만 요청하고 checkout 자격 증명을 남기지 않는다.
`pull_request_target`이나 외부 비밀 키 없이 체크아웃한 변경을 검사한다.

1. 프로젝트 계약, 경로, 규칙 ID와 기존 동결 증거 13개의 해시를 확인한다.
2. 세션 검사의 회귀 테스트를 실행한다.
3. 역할/작업/브랜치의 일치와 중복 브랜치, 종료 세션의 인수인계 누락을 확인한다.
4. 인수인계의 역할·상태·기준 커밋·버전·산출물 경로를 확인하고 Actions job summary에 세션 표를 남긴다.

역할별 PR은 S0 브랜치를 대상으로 열고 S0가 실제 diff와 관련 검증을 확인한 뒤 통합한다.
공통 변경은 S0의 별도 PR로 `docs/project-contract-and-session-roles`에 제안한다.
워크플로가 아직 없는 작업 브랜치는 기존 Project contract 검사만 실행할 수 있다.
새 워크플로를 포함한 PR 병합 트리 또는 작업 브랜치에서 Session collaboration 실행 여부를 확인해야 한다.
이벤트 검사는 새 에이전트를 생성하거나 커밋·병합·H1 승인·최종 평가를 자동 수행하지 않는다.

레지스트리의 `contract_version`은 각 작업이 끝난 브랜치의 계약 버전이다.
계약 버전이 올라가도 과거 인수인계를 다시 작성하지 않는다. `historical_contract_versions`에
명시적으로 남긴 버전은 그 버전으로 메타데이터를 검사하며, 새 의미 계약 충족을 인증하는 것은 아니다.
`result_commit`과 PR URL은 탐색용 기록이다. 오프라인 검사는 원격 PR 상태나 Git 계보를 검증하지 않는다.

## 외부 웹훅의 현재 상태

GitHub App으로 `GET /repos/LeeDongHyun00/MyLittleLLM/hooks`를 조회했으나
`403 Resource not accessible by integration`이 반환됐다. 수신 URL도 제공되지 않았다.
따라서 **외부 repository webhook은 연결하지 않았다**. 위 자동화는 GitHub Actions의 네이티브 이벤트다.
이는 자동 승인 심사의 거절이 아니라 GitHub 연결의 웹훅 관리 권한 제한이다.

외부 수신처가 정해지면 webhook 관리 권한이 있는 연결과 HTTPS endpoint를 사용해 별도 연결해야 한다.
그 경우 서명 검증(`X-Hub-Signature-256`), delivery ID 중복 방지, 허용 저장소/이벤트/브랜치와
실행 큐를 검토한다. 키나 URL을 지어내거나 공개 저장소에 비밀을 기록하지 않는다.

## 다음 대화에서 재개

```bash
git fetch origin
git switch s0/i01-session-collaboration
python scripts/check_project_contract.py
python scripts/check_session_handoffs.py --branch s0/i01-session-collaboration
```

브랜치가 로컬에 없으면 origin에서 새 worktree를 만들고 AGENTS → 계약 → STATE → 작업 카드 →
`latest_handoff` 순서로 읽는다. 새 세션은 자신에게 허용된 파일과 입력 산출물을 재확인한다.
세션 검사 통과는 원문 의미·정답·H1 독립 검수·E02 실행 준비가 완료됐다는 뜻이 아니다.

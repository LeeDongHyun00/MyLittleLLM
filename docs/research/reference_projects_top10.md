# 유사 프로젝트와 운영 방식 참고 TOP 10

조사일 **2026-11-04**. 이 순위는 MyLittleLLM의 입시·장학 문서 상담, 저사양 실행,
검증 가능한 규칙, 평가, 세션 분리에 도움이 되는 **참고 우선순위**다.
한국 입시 상담의 성공률이나 전 세계 프로젝트의 객관적 순위를 측정한 것이 아니다.

## 선정 방법과 성공도의 의미

GitHub에서 `local document RAG Qwen`, `RAG document parsing`, `policy rules as code`,
`LLM evaluation regression testing`, `spec driven development`를 검색하고 알려진 성숙 후보를 추가했다.
13개 후보의 공식 저장소 메타데이터·README·기본 브랜치 커밋·릴리스를 비교했다.
선정 10개는 당시 커밋에 고정한 README와 테스트 워크플로 정의도 확인했다.
Haystack 파이프라인 코드, OpenFisca 파라미터/패키지 설정은 추가로 읽었다.

평가 기준은 (1) 문서/로컬 LLM/조건 판정의 기능 유사성, (2) 현재 실패와 세션 문제에 대한 적용성,
(3) 공개 유지보수 근거다. 기능/컴포넌트 8개와 운영 도구 2개를 포함하는 편집 순위이며
검증되지 않은 가중치나 성공 확률을 만들지 않았다. 별이 적은 OpenFisca도 정책 계산과 직접 관련되어 높게 선정했다.

**성공도 대신 확인 가능한 성숙도 근거를 제시한다.** 스타·포크는 관심 지표이며 사용률·정확도·사업 성과가 아니다.
CI는 정의 파일을 읽었을 뿐, 직접 테스트를 실행하거나 최신 CI 통과를 확인하지 않았다.
README의 도입/사용자 수는 작성자 주장과 독립 검증을 구분하며 순위 점수로 쓰지 않았다.
우리 4 CPU·16GiB 환경에서 이 프로젝트들을 실행한 벤치마크도 아니다.

정확한 조회 수치, 기본 브랜치 SHA, 고정 README/워크플로 URL, 제외 후보와 검색 기록은
[JSON 스냅샷](../../data/research/reference_projects_snapshot.json)에 저장했다. 아래 날짜는 기본 브랜치 커밋 기준이며
아무 브랜치나 변경될 수 있는 `pushed_at`과 구분했다.

## 순위

| 순위 | 프로젝트 / 종류 | 우리와의 유사성 | 공개 관심·유지보수 관측 | 가져올 것 |
|---:|---|---|---|---|
| 1 | [deepset-ai/haystack](https://github.com/deepset-ai/haystack) / RAG 프레임워크 | 높음 | ★ 26,924; 기본 브랜치 2026-11-03; v2.31.1 | 검색·근거·생성 컴포넌트를 분리하고 연결 타입을 검증하는 구조 |
| 2 | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) / 문서 RAG 제품 | 높음 | ★ 93,464; 기본 브랜치 2026-11-04; v1.0.0-rc3 | 문서 파싱, 검색 구간 미리보기, 추적 가능한 인용 |
| 3 | [openfisca/openfisca-core](https://github.com/openfisca/openfisca-core) / 정책 규칙 엔진 | 부분 기능에 매우 높음 | ★ 252; 기본 브랜치 2026-10-23; 패키지 46.0.2 (pyproject.toml) | 기간별 정책 파라미터·계산 엔진과 국가별 규칙 패키지 분리 |
| 4 | [docling-project/docling](https://github.com/docling-project/docling) / 문서 파싱 도구 | 전처리에 높음 | ★ 70,088; 기본 브랜치 2026-11-03; v2.146.0 | PDF 레이아웃·표·OCR·통일 문서 표현으로 쪽/각주 맥락 보존 |
| 5 | [run-llama/llama_index](https://github.com/run-llama/llama_index) / 문서/검색 프레임워크 | 높음 | ★ 52,798; 기본 브랜치 2026-11-03; v0.14.25 | 문서 수집·검색 단위·메타데이터·질의/평가 구성 |
| 6 | [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) / LLM 평가 도구 | 평가에 매우 높음 | ★ 26,811; 기본 브랜치 2026-11-03; 0.125.1 | 선언적 사례·provider 비교·회귀 평가·CI |
| 7 | [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm) / 로컬 문서 상담 제품 | 높음 | ★ 67,911; 기본 브랜치 2026-11-04; v1.19.0 | 로컬 모델·OpenAI 호환 공급자·문서 workspace·출처 표시 |
| 8 | [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) / 문서 QA 제품 | 높음 | ★ 25,855; 기본 브랜치 2026-05-30; v0.12.0 | 로컬 모델, 하이브리드 검색·재순위화, PDF 인용 강조와 낮은 관련성 안내 |
| 9 | [github/spec-kit](https://github.com/github/spec-kit) / 명세 중심 개발 도구 | 세션 운영에 직접적 | ★ 146,815; 기본 브랜치 2026-11-03; v1.3.0 | 프로젝트 constitution → 명세 → 계획 → 작업 → 구현의 저장소 기록 |
| 10 | [Fission-AI/OpenSpec](https://github.com/Fission-AI/OpenSpec) / 변경 명세 관리 도구 | 세션 운영에 직접적 | ★ 73,861; 기본 브랜치 2026-11-02; v1.16.0 | 변경별 proposal/specs/design/tasks와 완료 후 archive |

## 프로젝트별 판단

### 1. deepset-ai/haystack

**적용 역할: S0/S2/S4.** 검색·근거·생성 컴포넌트를 분리하고 연결 타입을 검증하는 구조. Python 파이프라인·연결 검증·CI 정의를 공개하고 지속 릴리스.
한계와 선택: 전체 프레임워크 교체 없이 현재 함수의 I/O 계약부터 적용한다. 한국어 입시 정확도 검증은 별도다.
관측 라이선스: Apache-2.0. [고정 README](https://github.com/deepset-ai/haystack/blob/82dce4f7aed816916646542b8ab2c6c10ff46ae4/README.md) · [테스트 정의](https://github.com/deepset-ai/haystack/blob/82dce4f7aed816916646542b8ab2c6c10ff46ae4/.github/workflows/tests.yml) · [릴리스](https://github.com/deepset-ai/haystack/releases/tag/v2.31.1).

### 2. infiniflow/ragflow

**적용 역할: S1/S2.** 문서 파싱, 검색 구간 미리보기, 추적 가능한 인용. 활발한 기본 브랜치, 공개 테스트 정의와 릴리스 후보.
한계와 선택: 공식 요구 자원은 4 CPU·16GB·50GB이며 로컬 LLM 자원은 추가다. Qwen과 함께 현재 환경에 적합하다고 보장하지 않는다. v1.0.0-rc3는 API prerelease=false여도 태그상 RC다.
관측 라이선스: Apache-2.0. [고정 README](https://github.com/infiniflow/ragflow/blob/6aeb92deff2b46303559cd4c84a7b8db35ee3fb4/README.md) · [테스트 정의](https://github.com/infiniflow/ragflow/blob/6aeb92deff2b46303559cd4c84a7b8db35ee3fb4/.github/workflows/sep-tests.yml) · [릴리스](https://github.com/infiniflow/ragflow/releases/tag/v1.0.0-rc3).

### 3. openfisca/openfisca-core

**적용 역할: S3.** 기간별 정책 파라미터·계산 엔진과 국가별 규칙 패키지 분리. OpenFisca-France 연동과 기간별 파라미터 테스트가 공개됨.
한계와 선택: 대화형 LLM이 아니며 한국 입시 규칙도 제공하지 않는다. 개념을 참조하며 코드를 도입하면 AGPL-3.0-or-later 조건을 별도 검토한다.
관측 라이선스: AGPL-3.0-or-later (pyproject.toml). [고정 README](https://github.com/openfisca/openfisca-core/blob/db0484b54871b0a73e673478b3153841dba908e9/README.md) · [테스트 정의](https://github.com/openfisca/openfisca-core/blob/db0484b54871b0a73e673478b3153841dba908e9/.github/workflows/push.yaml) · [패키지 설정](https://github.com/openfisca/openfisca-core/blob/db0484b54871b0a73e673478b3153841dba908e9/pyproject.toml).

### 4. docling-project/docling

**적용 역할: S1.** PDF 레이아웃·표·OCR·통일 문서 표현으로 쪽/각주 맥락 보존. 지속 릴리스, 로컬 실행 지원과 CI 정의.
한계와 선택: 한국어 모집요강 표·각주 품질과 CPU 메모리는 우리 페이지로 측정해야 한다. 코드 MIT와 모델별 이용 조건은 구분한다.
관측 라이선스: MIT. [고정 README](https://github.com/docling-project/docling/blob/0634e2f74561e5950d69b2cace67ef95400506ad/README.md) · [테스트 정의](https://github.com/docling-project/docling/blob/0634e2f74561e5950d69b2cace67ef95400506ad/.github/workflows/ci.yml) · [릴리스](https://github.com/docling-project/docling/releases/tag/v2.146.0).

### 5. run-llama/llama_index

**적용 역할: S1/S2/S5.** 문서 수집·검색 단위·메타데이터·질의/평가 구성. OSS 프레임워크·통합·단위 테스트 정의와 릴리스.
한계와 선택: 현재 README는 회사 중심이 LlamaParse·liteparse·벤치마크로 이동했다고 명시한다. 유료 서비스와 OSS 프레임워크를 구분하며 클라우드 사용을 전제하지 않는다.
관측 라이선스: MIT. [고정 README](https://github.com/run-llama/llama_index/blob/444fdf0684707de62ebb447d658dba137dd9cb99/README.md) · [테스트 정의](https://github.com/run-llama/llama_index/blob/444fdf0684707de62ebb447d658dba137dd9cb99/.github/workflows/unit_test.yml) · [릴리스](https://github.com/run-llama/llama_index/releases/tag/v0.14.25).

### 6. promptfoo/promptfoo

**적용 역할: S5.** 선언적 사례·provider 비교·회귀 평가·CI. 평가 도구 릴리스, 테스트/CI와 문서 공개.
한계와 선택: 현재 Python 평가를 바로 교체하지 않는다. 로컬 실행이라도 외부 provider를 쓰면 요청이 전송될 수 있다. 자동 assertion은 전문가 gold를 대체하지 않는다.
관측 라이선스: MIT. [고정 README](https://github.com/promptfoo/promptfoo/blob/f2b012e3c9f9ffb7b23edf4cf88ea63febf72f5a/README.md) · [테스트 정의](https://github.com/promptfoo/promptfoo/blob/f2b012e3c9f9ffb7b23edf4cf88ea63febf72f5a/.github/workflows/main.yml) · [릴리스](https://github.com/promptfoo/promptfoo/releases/tag/0.125.1).

### 7. Mintplex-Labs/anything-llm

**적용 역할: S0/S4.** 로컬 모델·OpenAI 호환 공급자·문서 workspace·출처 표시. desktop/self-host 경로와 테스트 정의, 활발한 릴리스.
한계와 선택: workspace 분리 패턴을 대학/연도 범위 설계에 참고한다. 제품을 설치하면 우리 범위/자격 규칙이 자동 보장되는 것은 아니다.
관측 라이선스: MIT. [고정 README](https://github.com/Mintplex-Labs/anything-llm/blob/eda0eb5e25a4facbe2ea92d86612a39618bcb54c/README.md) · [테스트 정의](https://github.com/Mintplex-Labs/anything-llm/blob/eda0eb5e25a4facbe2ea92d86612a39618bcb54c/.github/workflows/run-tests.yaml) · [릴리스](https://github.com/Mintplex-Labs/anything-llm/releases/tag/v1.19.0).

### 8. Cinnamon/kotaemon

**적용 역할: S2/S4.** 로컬 모델, 하이브리드 검색·재순위화, PDF 인용 강조와 낮은 관련성 안내. 실행 가능한 문서 QA와 단위 테스트 정의, v0.12.0 릴리스.
한계와 선택: 기본 브랜치 최근 커밋은 2026-05-30으로 다른 후보보다 활동이 느리다. 우리 기존 하이브리드 검색의 퇴보를 고려해 별도 비교 없이 채택하지 않는다.
관측 라이선스: Apache-2.0. [고정 README](https://github.com/Cinnamon/kotaemon/blob/9ad3e4e49aa35b8acddd235918a5d9753c1cfdf9/README.md) · [테스트 정의](https://github.com/Cinnamon/kotaemon/blob/9ad3e4e49aa35b8acddd235918a5d9753c1cfdf9/.github/workflows/unit-test.yaml) · [릴리스](https://github.com/Cinnamon/kotaemon/releases/tag/v0.12.0).

### 9. github/spec-kit

**적용 역할: S0/전체.** 프로젝트 constitution → 명세 → 계획 → 작업 → 구현의 저장소 기록. 공개 템플릿/워크플로·테스트 정의·지속 릴리스.
한계와 선택: 입시/RAG 제품이 아니다. 장기 계약과 짧은 시작 문맥 패턴만 반영한다. 도구 설치와 전체 템플릿 도입은 하지 않았다.
관측 라이선스: MIT. [고정 README](https://github.com/github/spec-kit/blob/a227e38315cd38172b04f2fb8f06ce8b69b38331/README.md) · [테스트 정의](https://github.com/github/spec-kit/blob/a227e38315cd38172b04f2fb8f06ce8b69b38331/.github/workflows/test.yml) · [릴리스](https://github.com/github/spec-kit/releases/tag/v1.3.0).

### 10. Fission-AI/OpenSpec

**적용 역할: S0/전체.** 변경별 proposal/specs/design/tasks와 완료 후 archive. 변경 단위 파일 흐름·CI 정의·지속 릴리스.
한계와 선택: 입시 상담 정확도의 성공 근거는 아니다. 한 세션 한 작업 카드와 결정/인수인계 기록으로 축소 적용한다.
관측 라이선스: MIT. [고정 README](https://github.com/Fission-AI/OpenSpec/blob/2bd5bc3be964404657439f538857fcdf550f7398/README.md) · [테스트 정의](https://github.com/Fission-AI/OpenSpec/blob/2bd5bc3be964404657439f538857fcdf550f7398/.github/workflows/ci.yml) · [릴리스](https://github.com/Fission-AI/OpenSpec/releases/tag/v1.16.0).

## 이번 계약에 반영한 것

| 참고 | 지금 반영한 파일/절차 | 적용 상태 |
|---|---|---|
| Spec Kit | [프로젝트 계약](../../PROJECT_CONTRACT.md), [시작점](../../AGENTS.md), [결정 기록](../project/DECISIONS.md) | 저장소 계약·짧은 시작 문맥 도입 |
| OpenSpec | [작업 카드](../project/TASKS.md), [세션 역할](../project/SESSIONS.md), [인수인계 양식](../handoffs/TEMPLATE.json) | 한 세션 한 작업·종료 기록 도입 |
| Haystack | [인터페이스](../project/INTERFACES.md) | 기존 생산자/소비자 경계 문서화; 프레임워크 전환 아님 |
| OpenFisca | 규칙의 적용 연도/학기·원문 버전·부분 판정 계약 | 기존 Z3 구현 유지; 국가 패키지/엔진 설치 안 함 |
| promptfoo | 평가 세션 분리, 동결 기록·회귀·새 시험의 구분 | 기존 Python 평가 유지; 새 평가 프로토콜은 E01/E02 |
| Docling / RAGFlow | D01 원문/표/각주 확인 → R01 검색 조항 적중 | 다음 작업 카드에만 반영; 새 파서 성능은 미측정 |
| LlamaIndex / AnythingLLM / Kotaemon | 근거 객체·scope 분리·인용 확인·실패 표시 | 인터페이스 참고; 새 UI/검색기 설치 안 함 |

큰 스택을 한꺼번에 설치하는 대신 지금 구현의 경계를 기록한다. 다음 변경은 D01/R01/M01의 작은 비교로 검증한다.
표·질의·검색·생성을 동시에 바꿔 한 점수를 얻으면 어느 변경이 효과를 냈는지 알 수 없다.

## 검토했지만 이번 10개에서 제외

- [onyx-dot-app/onyx](https://github.com/onyx-dot-app/onyx): 문서 검색 유사성은 높으나 enterprise 스택 비중이 크다. Lite 모드는 문서 색인을 지원하지 않아 현재 저사양 문서 RAG 대안으로 단정할 수 없다. CE MIT와 EE 조건도 분리해야 한다.
- [open-webui/open-webui](https://github.com/open-webui/open-webui): 로컬 모델 UI로 유용하지만 이번 규칙·평가·세션 계약에 대한 직접성은 선정 후보보다 낮다. 현재 라이선스의 브랜딩 조건을 단순 MIT로 표시할 수 없다.
- [CatalaLang/catala](https://github.com/CatalaLang/catala): 법규 예외의 형식화와 문서/코드 생성은 유사하지만 별도 DSL 도입 비용이 크다. 현재 Z3 규칙을 유지하면서 OpenFisca의 기간/파라미터 관리 패턴을 우선 참조한다.

이 목록에는 한국 대학 입학 공정성·진학 결과를 독립 실험으로 입증한 시스템이라는 주장을 붙이지 않았다.
모든 관련 프로젝트를 망라한 검색도 아니다. 다음 갱신에서는 최신 상태뿐 아니라 우리 실패군에 대한 실제 비교 결과를 추가한다.

# D01 — 월 상한 근거와 추출/검색 경로 점검

2026-11-04 · S1 · **부분 완료**. 별도 배포본의 근거 확인과 재현 자료는 준비했다.
기존 제주대 PDF 재확보·해시 재확인과 공식 최신 수정본 대조는 남아 있다.
[원문 manifest](../data/development/d01/source_manifest.json),
[개발 fixture](../data/development/d01/clauses.json),
[실행 결과](d01_source_audit.json)를 함께 사용한다.

## 확인된 결과

확보한 [동서대 배포 PDF](https://www.dongseo.ac.kr/kr/index.php?pCode=MN2000195&mode=fdn&idx=102875&num=1)에서
월 상한 조항은 **PDF 10쪽(인쇄 8쪽)**에 있으며, pypdf plain/layout 추출과 현재 청크에도 남는다.
공개 V12를 현재 BM25로 조회하면 23쪽, 3쪽, 10쪽 순서다. top-2에서 핵심 근거가 빠진다.
이 배포본에서는 숫자가 추출 단계에서 사라진 것이 아니라 **검색 순위에서 밀린 것**으로 확인했다.

| 구간 | 확인한 내용/손실 | 후속 작업 |
|---|---|---|
| PDF 10쪽 금액·각주 | 상한, 첫 달 예외, 인정 비용 및 중복 청구 제한 텍스트가 유지됨 | 금액과 연결 각주를 함께 제공 |
| PDF 10쪽 `c0` | 상한 뒤의 각주 중간에서 잘림 | 상한 문자열만 있으면 근거 충분으로 처리하지 않음 |
| PDF 10쪽 `c740` | 금액·첫 달·비용 예외 표지어를 함께 보존. V12 순위 3위 | R01은 실제 검색으로 이 조항을 찾는지 비교 |
| PDF 11쪽(인쇄 9쪽) | 기간·방학 예외와 학교별/개인별 한도 표 | 두 한도를 별도 조건으로 연결 |
| PDF 10·11쪽 표 | plain은 셀 경계 소실. layout은 단순 표를 개선하지만 병합 셀 관계를 보장하지 않음 | 렌더 대조 및 조항/각주 연결 필요 |
| PDF 26쪽 | FAQ 표지 시작을 렌더로 확인 | 기존 1–25쪽 범위와 분리; 26–123쪽은 이번 개발 주 색인에서 제외 |

표지어 존재 검사는 의미 검수가 아니다. 수동 표 대응과 좌표 범위는 **AI 확인 후보**이며
사람 승인 gold가 아니다. 전체 문서의 표·각주를 전수 확인한 결과도 아니다.
원문 10·11·26쪽을 렌더와 대조했으며 이미지/전체 텍스트는 Git 제외 outputs에만 보관했다.

## 원문 버전과 남은 차이

| 자료 | SHA-256 앞부분 / 크기 | 이번 확인 |
|---|---|---|
| 기존 제주대 snapshot | `8e4c1f6b…` / 2,375,263 bytes | 다운로드 HTTP 503, upstream 인증서 검증 오류. 과거 장부의 해시만 있음 |
| 동서대 재게시본 | `22af5aaa…` / 2,396,246 bytes, 123쪽 | TLS 검증을 유지한 HTTPS 다운로드·로컬 SHA 확인, 별도 source ID 사용 |
| 영남대 자료 | `5192afa8…` / 150,491 bytes, 8쪽 | 해당 대학 자체 계획이므로 원본 대체에서 제외 |

동서대 **게시글은 2학기 안내지만 첨부 표지는 2026년 1학기·2026.2**다.
[게시글](https://www.dongseo.ac.kr/kr/index.php?idx=102875&mode=view&pCode=MN2000195)도
1학기 자료를 첨부했다고 명시한다. 따라서 source scope는 2026년 **1학기**이며 2학기로 확장하지 않는다.
일치하는 쪽 배치와 검색 결과만으로 두 PDF가 의미상 완전히 같다고 판단하지 않았다.

[재단 공지](https://www.kosaf.go.kr/ko/notice.do?mode=view&page=1&seqNo=20660)의
2026-03-13 안내를 확인했지만 첨부 바이트는 확보하지 못했다.
정정/수정 검색에서 대체 문서를 확정하지 못했으며, **최신 수정본 확인 완료로 쓰지 않는다**.
이 제한 때문에 원래 고정 실행의 원문·전체 색인을 재현했다고 주장할 수 없다.

## 실행 범위와 자원

코드 `fdb98e8f042dc0aefc774970821d832a8c4df346`의
[감사 스크립트](../scripts/audit_d01_source.py)로 기존 `build_index`와 `retrieve`를 호출했다.
housing scope에 들어가는 소스만 별도 root에 복원했다. 원문 1–28쪽을 추출하고 기존 설정과 같이
1–25쪽만 색인했다(24쪽, 38청크). 2쪽 목차도 기존 조건대로 포함했다.
정식 새 실험에서 목차를 제외할지는 사전에 결정하며 이번 진단에서는 바꾸지 않았다.

Python 3.12.14, pypdf 6.10.0, 렌더 PyMuPDF 1.26.6으로 실행했다.
측정 시간 2.764초, 프로세스 peak RSS 74,404 KiB, 모델 호출 0회다.
이는 한 소스 감사의 측정치이며 서비스 지연/색인 전체 자원 추정치가 아니다.
Docling은 미설치이며 이번 확인에 필요하지 않아 도입하지 않았다. Docling 성능 비교를 했다는 뜻은 아니다.

확인 결과 원문 숫자 삭제와 청크 경계/표 구조 손실, 검색 순위 문제를 구별할 수 있었다.
이 작업에서 검색 방법·k·프롬프트·모델 설정을 변경하거나 V12 생성 응답을 다시 만들지는 않았다.
검색 누락이 고쳐졌을 때 생성 반복/잘림도 해결될지는 M01에서 별도로 확인해야 한다.

## 재현 및 재개

현재 환경의 다운로드 파일로 **새 경로**에 실행한다. pypdf는 requirements-data.txt의 버전을 사용한다.
`--render`는 PyMuPDF가 설치된 경우만 사용하며 생략해도 텍스트/검색 감사는 실행할 수 있다.

```bash
python scripts/audit_d01_source.py \
  --pdf outputs/d01-20261104-source-audit/22af5aaa41f0d32377f4c5adbe682de502f1e245712cf44432e448e312b83aa2.pdf \
  --source-version dongseo_republication \
  --output outputs/d01-next-mirror-audit --render
python -m unittest discover -s tests -p test_d01_audit.py -v
python -m unittest discover -s tests -p test_runtime_logic.py -v
python scripts/check_project_contract.py
git diff --check
```

다른 환경에서는 아래 공식 재게시 링크에서 다시 받고 감사 시 manifest 해시 일치를 요구한다.
기존 폴더에 덮어쓰지 않도록 `mkdir` 성공 후 다운로드한다.

```bash
mkdir outputs/d01-new-download && curl --fail --max-time 45 \
  'https://www.dongseo.ac.kr/kr/index.php?pCode=MN2000195&mode=fdn&idx=102875&num=1' \
  --output outputs/d01-new-download/source.pdf
```

원래 제주대 파일을 확보하면 `--source-version historical_jeju`로 **또 다른 출력 경로**에 실행한다.
입력 바이트가 원래 SHA와 다르면 스크립트가 거부한다. mirror fixture의 좌표/조항 대응은
원본에서 새로 시각 검토해야 하며 자동 승계하지 않는다. 원문이 갱신됐으면 새 version을 등록하고
기존 hash/결과를 보존한다.

## 소비자 호환성과 인수인계

`paired_clause_ids`, `context_bbox_pt`, `manually_transcribed_pairs`는 D01 sidecar의 제안 필드다.
기존 검색 chunk, facts, 규칙/계산 입력, corpus 설정에는 새 필드를 주입하지 않았다.
S2는 이 자료를 개발 평가의 근거 충분성 검사에 사용할 수 있지만 질문별 정답 숫자/쪽/ID를
검색 로직에 하드코딩하지 않는다. mirror source ID를 기존 규칙의 원문 hash로 위장하지 않는다.

R01은 **별도 배포본의 공개 개발 비교**부터 진행할 수 있다. 기존 8개 문서의 전체 색인 복원이나
원래 V12 실행의 완전 재현이 필요한 비교는 원본 확보를 기다린다.
원문 정정/최신성 확인, 두 버전의 차이 검토, 독립 H1은 남아 있다. E02 최종 평가는 계속 대기다.
STATE/TASKS와 공통 계약 채택은 S0가 통합하며 기존 점수와 평가 기록은 그대로 보존한다.

# 모델 환경 확인과 직접 설치

## 직접 설치 완료 (2026-11-02 후속 작업)

사용자의 직접 설치 요청에 따라 이번 환경에 공식 Qwen3-4B Q4_K_M GGUF와 llama.cpp CPU 서버를 설치했다. 아래 경로에 실제 파일이 있고 `/v1/models`와 추론 요청을 확인했다.

```text
/workspace/MyLittleLLM/models/Qwen3-4B-Q4_K_M.gguf
/workspace/MyLittleLLM/serve.sh
/workspace/MyLittleLLM/chat.py
http://127.0.0.1:8080/v1
```

설치 버전·검증한 체크섬·실행 명령은 [설치 및 실행](running.md)에 기록한다. 이전에 사용자가 보고한 별도 환경의 설치를 가져온 것이 아니라 공식 배포본을 새로 설치했다. 모델 추론과 RAG 예비 비교를 수행하며, 미세조정은 수행하지 않았다.

## 최초 확인 기록 (설치 전)

확인일: 2026-11-02. 사용자가 제공한 실행 정보:

```text
모델: /workspace/MyLittleLLM/models/Qwen3-4B-Q4_K_M.gguf
서버: http://127.0.0.1:8080/v1
API 모델명: Qwen3-4B
실행: /workspace/MyLittleLLM/serve.sh
대화: python /workspace/MyLittleLLM/chat.py
```

이번 대화에 실제 연결된 환경에서는 해당 모델, `serve.sh`, `chat.py`가 확인되지 않았고 `/v1/models` 요청은 연결 거절이었다. 저장소도 처음 확인할 때 빈 상태였다. 별도 실행 환경/세션에 설치되어 있을 가능성이 있으며 사용자 측 설치가 없다고 단정하지 않는다. 위 파일을 임의로 만들거나 기존 설치를 덮어쓰지 않았다.

최초 수집 단계에서는 모델 학습, 답변 생성, 정확도 측정을 수행하지 않았고 데이터 수집과 분할 코드만 검증했다. 이 상태는 위 후속 직접 설치로 변경됐다. `localhost`는 서버가 실행되는 환경 내부에서만 의미가 있다.

모델이 있는 환경에서 확인:

```bash
python scripts/probe_model.py --base-url http://127.0.0.1:8080/v1
```

연결되면 모델 파일 해시와 llama.cpp 서버 버전, GPU/CPU·RAM, 컨텍스트 길이, thinking 설정을 함께 기록한다. `/v1/models`의 별칭만으로 원본 모델 변형을 확정할 수 없다.

Q4_K_M GGUF는 이번 RAG 추론 비교에 사용할 수 있다. 통상적인 LoRA/QLoRA 미세조정에는 대응하는 원본 모델 가중치와 별도 학습 소프트웨어가 필요하다. 미세조정 후에도 최종 배포 양자화에서 다시 정확도를 평가한다.

# Langflow A2A Components

Langflow용 A2A (Agent-to-Agent) 컴포넌트 모음입니다.

## 컴포넌트

### 1. A2A SSE Proxy Launcher

Langflow flow를 A2A JSON-RPC/SSE 프록시로 노출하는 컴포넌트입니다.

#### 주요 기능
- **JSON-RPC over HTTP**: POST `/` 엔드포인트 제공
- **Server-Sent Events (SSE)**: 실시간 스트리밍 지원
- **자동 Flow 해석**: flow_id, flow_name, 또는 자동 선택
- **A2A Discovery 서비스 연동**: 프록시 시작 시 자동 에이전트 등록

#### 입력 파라미터

**기본 설정**
- `enabled`: 프록시 활성화 여부
- `host`: 바인딩 호스트 (기본값: 0.0.0.0)
- `port`: 바인딩 포트 (기본값: 5008)

**Langflow 연결**
- `langflow_url`: Langflow 서버 URL
- `langflow_api_key`: Langflow API 키
- `flow_name`: Flow 이름 (우선순위: flow_id > flow_name > 자동선택)
- `flow_id`: Flow ID (선택사항)

**스트리밍 설정**
- `stream_path`: 스트리밍 경로 형식
- `prefer_session_as_flow`: session_id를 flow_id로 사용 여부
- `auto_pick_singleton`: 단일 flow 자동 선택 여부

**A2A Discovery 서비스**
- `discovery_url`: A2A Discovery 서비스 URL (예: http://localhost:8000)
- `auto_register_discovery`: 자동 등록 활성화 여부
- `agent_card`: Agent Card 정보 (Discovery 서비스 등록용)

#### A2A Discovery 서비스 연동

프록시가 시작되면 자동으로 A2A Discovery 서비스에 에이전트를 등록합니다:

1. **에이전트 등록**: POST `/agent` 엔드포인트 호출
2. **자동 상태 관리**: Discovery 서비스가 3초마다 자동으로 `/ping`을 호출하여 상태를 "active"/"inactive"로 자동 변경
3. **자동 정리**: 컴포넌트 소멸 시 로그 기록 (Discovery 서비스가 자동으로 상태를 확인)

#### API 엔드포인트

- `GET /ping`: Discovery 서비스용 간단한 헬스체크 (HTTP 200만 확인)
- `GET /health`: 상세한 헬스체크 (flow 상태 포함)
- `GET /agent-card`: Agent Card 정보
- `GET /.well-known/agent-card.json`: 표준 Agent Card 경로
- `POST /`: JSON-RPC 요청 처리

### 2. Agent Card (A2A)

A2A 호환 Agent Card JSON을 생성하는 컴포넌트입니다.

#### 주요 기능
- **표준 A2A 형식**: A2A 프로토콜에 맞는 Agent Card 생성
- **스킬 관리**: 텍스트 또는 HandleInput을 통한 스킬 정의
- **능력 설정**: 스트리밍, 푸시 알림, 상태 전환 히스토리 등

#### 입력 파라미터

**필수 정보**
- `name`: 에이전트 이름
- `url`: 에이전트 기본 URL

**고급 설정**
- `description`: 에이전트 설명
- `version`: 버전 정보
- `capabilities`: 에이전트 능력 (스트리밍, 푸시 알림, 상태 히스토리)
- `preferred_transport`: 선호하는 전송 프로토콜
- `default_input_modes`: 기본 입력 모드
- `default_output_modes`: 기본 출력 모드
- `skills_handle`: 스킬 컴포넌트 연결
- `extra`: 추가 필드 (JSON 형식)

### 3. Skill

에이전트의 스킬을 정의하는 컴포넌트입니다.

## 사용 예시

### 1. 기본 프록시 설정

```python
# A2A SSE Proxy Launcher 설정
proxy = ProxyLauncher(
    enabled=True,
    host="0.0.0.0",
    port=5008,
    langflow_url="http://127.0.0.1:7860",
    langflow_api_key="your-api-key",
    flow_name="My Flow",
    discovery_url="http://localhost:8000",
    auto_register_discovery=True
)
```

### 2. Agent Card 생성

```python
# Agent Card 생성
agent_card = AgentCardComponent(
    name="My Agent",
    url="http://localhost:5008",
    description="A sample A2A agent",
    version="1.0.0",
    cap_streaming=True,
    preferred_transport="JSONRPC"
)
```

### 3. Discovery 서비스 연동

프록시가 시작되면 자동으로:
1. A2A Discovery 서비스에 에이전트 등록
2. Discovery 서비스가 3초마다 자동으로 `/ping`을 호출하여 상태를 "active"로 변경
3. 프록시 종료 시 로그 기록 (Discovery 서비스가 자동으로 상태를 확인)

## A2A Discovery 서비스

이 컴포넌트는 [A2A-Agent-Matrix](https://github.com/dkdoc7/A2A-Agent-Matirix)의 Discovery 서비스와 연동됩니다.

### Discovery 서비스 API

- `POST /agent`: 에이전트 등록
- `GET /agents`: 에이전트 목록 조회
- **자동 상태 관리**: 3초마다 자동으로 `/ping`을 호출하여 상태를 자동 변경

## 요구사항

- Python 3.8+
- Langflow
- httpx
- uvicorn
- fastapi

## 설치

```bash
# 의존성 설치
pip install httpx uvicorn fastapi

# Langflow에 컴포넌트 추가
# langflow-a2a/ 디렉토리를 Langflow의 custom_components 경로에 복사
```

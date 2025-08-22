#!/usr/bin/env python3
"""
A2A Discovery 서비스 연동 기능 사용 예시
"""

import json
from a2a_proxy import ProxyLauncher
from agent_card import AgentCardComponent

def create_agent_card():
    """Agent Card 컴포넌트를 생성하고 설정합니다."""
    
    # Agent Card 생성
    agent_card = AgentCardComponent()
    
    # 기본 정보 설정
    agent_card._attributes = {
        "name": "My A2A Agent",
        "url": "http://localhost:5008",
        "description": "Langflow 기반 A2A 에이전트",
        "version": "1.0.0",
        "cap_streaming": True,
        "cap_push": False,
        "cap_state": False,
        "preferred_transport": "JSONRPC",
        "default_input_modes": "text,text/plain",
        "default_output_modes": "text,text/plain",
        "skills_handle": [],
        "extra": '{"owner": "my-team", "license": "MIT"}'
    }
    
    return agent_card

def create_proxy_launcher(agent_card):
    """Proxy Launcher 컴포넌트를 생성하고 설정합니다."""
    
    # Proxy Launcher 생성
    proxy = ProxyLauncher()
    
    # 기본 설정
    proxy._attributes = {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 5008,
        "langflow_url": "http://127.0.0.1:7860",
        "langflow_api_key": "your-api-key-here",
        "flow_name": "My Flow",
        "flow_id": "",
        "stream_path": "/api/v1/run/{flow_id}?stream=true",
        "prefer_session_as_flow": False,
        "auto_pick_singleton": True,
        "agent_card": agent_card,
        "discovery_url": "http://localhost:8000",
        "auto_register_discovery": True
    }
    
    return proxy

def main():
    """메인 실행 함수"""
    
    print("🚀 A2A Discovery 서비스 연동 예시")
    print("=" * 50)
    
    try:
        # 1. Agent Card 생성
        print("1️⃣ Agent Card 생성 중...")
        agent_card = create_agent_card()
        card_message = agent_card.build_card_message()
        print(f"✅ Agent Card 생성 완료: {card_message.text[:100]}...")
        
        # 2. Proxy Launcher 생성
        print("2️⃣ Proxy Launcher 생성 중...")
        proxy = create_proxy_launcher(agent_card)
        
        # 3. 프록시 시작 (실제로는 Langflow에서 실행됨)
        print("3️⃣ 프록시 시작 시뮬레이션...")
        print("   실제 사용 시에는 Langflow에서 이 컴포넌트를 실행하세요.")
        
        # 4. 설정 정보 출력
        print("\n📋 설정 정보:")
        print(f"   - 프록시 URL: http://localhost:5008")
        print(f"   - Discovery 서비스: http://localhost:8000")
        print(f"   - 자동 등록: 활성화")
        print(f"   - Agent 이름: {agent_card._attributes['name']}")
        
        # 5. 예상 동작 설명
        print("\n🔍 예상 동작:")
        print("   1. 프록시가 시작되면 자동으로 Discovery 서비스에 에이전트 등록")
        print("   2. 에이전트 상태가 'active'로 설정됨")
        print("   3. 프록시 종료 시 자동으로 에이전트 상태를 'inactive'로 변경")
        
        print("\n🎯 설정 완료! Langflow에서 이 컴포넌트를 사용하세요.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()

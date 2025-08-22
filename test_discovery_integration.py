#!/usr/bin/env python3
"""
A2A Discovery 서비스 연동 기능 테스트 스크립트
"""

import json
import time
from a2a_proxy import _register_agent_to_discovery, _deactivate_agent_in_discovery

def test_discovery_integration():
    """Discovery 서비스 연동 기능을 테스트합니다."""
    
    # 테스트 설정
    discovery_url = "http://localhost:8000"
    base_url = "http://localhost:5008"
    
    # 테스트용 Agent Card
    agent_card = {
        "name": "test-agent-001",
        "description": "테스트용 A2A 에이전트",
        "url": base_url,
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False
        },
        "defaultInputModes": ["text", "text/plain"],
        "defaultOutputModes": ["text", "text/plain"],
        "skills": [
            {
                "id": "skill-001",
                "name": "테스트 스킬",
                "description": "간단한 테스트 스킬",
                "tags": ["test", "demo"],
                "examples": ["테스트 입력"]
            }
        ]
    }
    
    print("🧪 A2A Discovery 서비스 연동 테스트 시작")
    print(f"Discovery URL: {discovery_url}")
    print(f"Agent URL: {base_url}")
    print(f"Agent Card: {json.dumps(agent_card, ensure_ascii=False, indent=2)}")
    print("-" * 50)
    
    # 1. 에이전트 등록 테스트
    print("1️⃣ 에이전트 등록 테스트...")
    try:
        success, message = _register_agent_to_discovery(
            discovery_url=discovery_url,
            agent_card=agent_card,
            base_url=base_url
        )
        
        if success:
            print(f"✅ 성공: {message}")
        else:
            print(f"❌ 실패: {message}")
            return False
            
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return False
    
    # 2. 잠시 대기 (등록 확인용)
    print("2️⃣ 등록 확인을 위해 잠시 대기...")
    time.sleep(2)
    
    # 3. 에이전트 상태 비활성화 테스트
    print("3️⃣ 에이전트 상태 비활성화 테스트...")
    try:
        success, message = _deactivate_agent_in_discovery(
            discovery_url=discovery_url,
            agent_card=agent_card
        )
        
        if success:
            print(f"✅ 성공: {message}")
        else:
            print(f"❌ 실패: {message}")
            return False
            
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return False
    
    print("-" * 50)
    print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
    return True

def test_discovery_service_health():
    """Discovery 서비스 상태를 확인합니다."""
    
    import httpx
    
    discovery_url = "http://localhost:8000"
    
    print("🏥 Discovery 서비스 상태 확인...")
    
    try:
        # 기본 정보 확인
        response = httpx.get(f"{discovery_url}/", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Discovery 서비스 활성: {info.get('protocol', 'Unknown')} v{info.get('version', 'Unknown')}")
            return True
        else:
            print(f"❌ Discovery 서비스 응답 오류: HTTP {response.status_code}")
            return False
            
    except httpx.ConnectError:
        print(f"❌ Discovery 서비스에 연결할 수 없습니다: {discovery_url}")
        print("   Discovery 서비스가 실행 중인지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 상태 확인 중 오류: {e}")
        return False

if __name__ == "__main__":
    print("🚀 A2A Discovery 서비스 연동 테스트")
    print("=" * 60)
    
    # Discovery 서비스 상태 확인
    if not test_discovery_service_health():
        print("\n⚠️  Discovery 서비스가 실행되지 않았습니다.")
        print("   다음 명령으로 Discovery 서비스를 시작하세요:")
        print("   cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        exit(1)
    
    print()
    
    # 연동 기능 테스트
    if test_discovery_integration():
        print("\n🎯 테스트 결과: 성공")
        exit(0)
    else:
        print("\n💥 테스트 결과: 실패")
        exit(1)

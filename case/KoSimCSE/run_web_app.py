#!/usr/bin/env python3
"""
사고 유사도 검색 시스템 웹 애플리케이션 실행 스크립트
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """필요한 패키지들이 설치되어 있는지 확인"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'plotly',
        'torch',
        'transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 다음 패키지들이 설치되지 않았습니다: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 모든 필요한 패키지가 설치되어 있습니다.")
    return True

def check_data_files():
    """필요한 데이터 파일들이 있는지 확인"""
    data_files = [
        'data/testy.csv'
    ]
    
    missing_files = []
    
    for file_path in data_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 다음 데이터 파일들이 없습니다: {', '.join(missing_files)}")
        print("testy.csv 파일을 data 폴더에 넣어주세요.")
        return False
    
    print("✅ 모든 필요한 데이터 파일이 있습니다.")
    return True

def run_web_app():
    """웹 애플리케이션 실행"""
    print("🚀 사고 유사도 검색 시스템을 시작합니다...")
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    print(f"📁 현재 디렉토리: {current_dir}")
    
    # 의존성 확인
    if not check_dependencies():
        return False
    
    # 데이터 파일 확인
    if not check_data_files():
        return False
    
    # Streamlit 앱 실행
    try:
        print("🌐 웹 브라우저에서 http://localhost:8501 을 열어주세요.")
        print("🔄 서버를 중지하려면 Ctrl+C를 누르세요.")
        
        # Streamlit 명령어 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_interface.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 웹 애플리케이션이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    run_web_app() 
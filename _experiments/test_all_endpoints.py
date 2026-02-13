"""
Comprehensive API Endpoint Testing Script
Tests all PrepGen AI Service endpoints with detailed output
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_FILE = "sample.txt"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_response(response):
    """Pretty print response"""
    try:
        data = response.json()
        print(f"{Colors.OKBLUE}{json.dumps(data, indent=2)}{Colors.ENDC}")
    except:
        print(f"{Colors.OKBLUE}{response.text}{Colors.ENDC}")

def test_health_check():
    """Test 1: Health Check Endpoint"""
    print_header("TEST 1: Health Check")
    
    try:
        print_info(f"GET {BASE_URL}/health")
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_response(response)
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_session_stats():
    """Test 2: Session Statistics"""
    print_header("TEST 2: Session Statistics")
    
    try:
        print_info(f"GET {BASE_URL}/sessions/stats")
        response = requests.get(f"{BASE_URL}/sessions/stats")
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_response(response)
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_file_upload():
    """Test 3: File Upload"""
    print_header("TEST 3: File Upload")
    
    try:
        # Check if test file exists
        if not Path(TEST_FILE).exists():
            print_warning(f"Test file '{TEST_FILE}' not found. Creating sample file...")
            with open(TEST_FILE, 'w', encoding='utf-8') as f:
                f.write("""
Machine Learning is a subset of artificial intelligence that focuses on building systems 
that can learn from and make decisions based on data. It involves training algorithms on 
large datasets to recognize patterns and make predictions. Common applications include 
image recognition, natural language processing, recommendation systems, and autonomous vehicles.

Deep Learning is a specialized branch of machine learning that uses neural networks with 
multiple layers. These networks can automatically learn hierarchical representations of data, 
making them particularly effective for complex tasks like computer vision and speech recognition.

The field has grown exponentially in recent years due to increased computing power, availability 
of large datasets, and advances in algorithms.
                """.strip())
            print_success(f"Created test file: {TEST_FILE}")
        
        print_info(f"POST {BASE_URL}/upload")
        print_info(f"Uploading file: {TEST_FILE}")
        
        with open(TEST_FILE, 'rb') as f:
            files = {'file': (TEST_FILE, f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_response(response)
            data = response.json()
            return data.get('session_id')
        else:
            print_error(f"Status Code: {response.status_code}")
            print_response(response)
            return None
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return None

def test_summarization(session_id):
    """Test 4: Summarization (with cache test)"""
    print_header("TEST 4: Summarization")
    
    if not session_id:
        print_error("No session_id available. Skipping test.")
        return False
    
    try:
        # First summarization (no cache)
        print_info(f"POST {BASE_URL}/summarize")
        print_info("First summarization (building cache)...")
        
        payload = {
            "session_id": session_id,
            "max_length": 150,
            "min_length": 50
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/summarize", json=payload)
        first_duration = time.time() - start_time
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_success(f"Duration: {first_duration:.2f}s")
            print_response(response)
            
            # Second summarization (with cache)
            print_info("\nSecond summarization (using cache)...")
            
            start_time = time.time()
            response2 = requests.post(f"{BASE_URL}/summarize", json=payload)
            second_duration = time.time() - start_time
            
            if response2.status_code == 200:
                print_success(f"Status Code: {response2.status_code}")
                print_success(f"Duration: {second_duration:.2f}s")
                
                speedup = first_duration / second_duration if second_duration > 0 else 0
                print_success(f"🚀 Cache Speedup: {speedup:.1f}x faster!")
                print_response(response2)
                return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_response(response)
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_quiz_generation(session_id):
    """Test 5: Quiz Generation"""
    print_header("TEST 5: Quiz Generation")
    
    if not session_id:
        print_error("No session_id available. Skipping test.")
        return False
    
    try:
        print_info(f"POST {BASE_URL}/quiz")
        
        payload = {
            "session_id": session_id,
            "num_questions": 3,
            "difficulty": "medium"
        }
        
        print_info("Generating quiz questions...")
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/quiz", json=payload, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_success(f"Duration: {duration:.2f}s")
            print_response(response)
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_response(response)
            return False
            
    except requests.exceptions.Timeout:
        print_error("Request timeout (quiz generation can take time)")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_ask_question(session_id):
    """Test 6: Ask Question (RAG)"""
    print_header("TEST 6: Ask Question (RAG)")
    
    if not session_id:
        print_error("No session_id available. Skipping test.")
        return False
    
    try:
        print_info(f"POST {BASE_URL}/ask")
        
        payload = {
            "session_id": session_id,
            "question": "What is machine learning?"
        }
        
        print_info(f"Question: {payload['question']}")
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/ask", json=payload, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_success(f"Duration: {duration:.2f}s")
            print_response(response)
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_response(response)
            return False
            
    except requests.exceptions.Timeout:
        print_error("Request timeout (RAG can take time)")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_youtube_summarization():
    """Test 7: YouTube Summarization"""
    print_header("TEST 7: YouTube Summarization")
    
    try:
        print_info(f"POST {BASE_URL}/summarize-youtube")
        
        payload = {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Sample URL
            "max_length": 150
        }
        
        print_info(f"YouTube URL: {payload['url']}")
        print_warning("This may fail if youtube-transcript-api is not installed")
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/summarize-youtube", json=payload, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_success(f"Duration: {duration:.2f}s")
            print_response(response)
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_response(response)
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_delete_session(session_id):
    """Test 8: Delete Session"""
    print_header("TEST 8: Delete Session")
    
    if not session_id:
        print_error("No session_id available. Skipping test.")
        return False
    
    try:
        print_info(f"DELETE {BASE_URL}/sessions/{session_id}")
        response = requests.delete(f"{BASE_URL}/sessions/{session_id}")
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_response(response)
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_response(response)
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     PrepGen AI Service - Comprehensive Endpoint Testing    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    print_info(f"Testing server at: {BASE_URL}")
    print_info(f"Test file: {TEST_FILE}\n")
    
    results = {}
    session_id = None
    
    # Test 1: Health Check
    results['health'] = test_health_check()
    time.sleep(1)
    
    # Test 2: Session Stats
    results['stats'] = test_session_stats()
    time.sleep(1)
    
    # Test 3: Upload
    session_id = test_file_upload()
    results['upload'] = session_id is not None
    time.sleep(1)
    
    # Test 4: Summarization (with cache)
    results['summarize'] = test_summarization(session_id)
    time.sleep(1)
    
    # Test 5: Quiz Generation
    results['quiz'] = test_quiz_generation(session_id)
    time.sleep(1)
    
    # Test 6: Ask Question
    results['ask'] = test_ask_question(session_id)
    time.sleep(1)
    
    # Test 7: YouTube Summarization
    results['youtube'] = test_youtube_summarization()
    time.sleep(1)
    
    # Test 8: Delete Session
    results['delete'] = test_delete_session(session_id)
    
    # Summary
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test, status in results.items():
        if status:
            print_success(f"{test.upper()}: PASSED")
        else:
            print_error(f"{test.upper()}: FAILED")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.ENDC}")
    
    if passed == total:
        print_success("\n🎉 All tests passed! Your API is working perfectly!")
    elif passed >= total * 0.7:
        print_warning(f"\n⚠️  Most tests passed ({passed}/{total}), but some failed.")
    else:
        print_error(f"\n❌ Many tests failed ({total-passed}/{total}). Check server logs.")
    
    print("\n")

if __name__ == "__main__":
    main()

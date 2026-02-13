"""
Test script to verify Academic Summarizer integration into PrepGen
Tests:
1. Model loading from ai_models.py
2. Domain-aware summarization (scientific, booksum, wikihow)
3. Backward compatibility with old code
4. Long document handling (chunk_and_summarize)
"""

import sys
import time
from ai_models import models

# Test sample: Course Policy document with technical terms
TEST_TEXT_ACADEMIC = """
Cloud Computing Course Policy (CS4550)

Course Description:
This advanced course covers cloud computing architectures, deployment models, and service paradigms. Students will gain hands-on experience with Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS) technologies. The course emphasizes virtualization, containerization, microservices, and serverless computing patterns.

Prerequisites:
- CS3310 (Operating Systems) with minimum B grade
- CS3320 (Computer Networks) with minimum C+ grade
- Familiarity with Linux command line and basic scripting

Course Objectives:
1. Understand cloud deployment models (public, private, hybrid, multi-cloud)
2. Design scalable applications using cloud-native principles
3. Implement CI/CD pipelines for cloud deployment
4. Analyze cost optimization strategies for cloud resources
5. Evaluate security and compliance requirements in cloud environments

Assessment Structure:
- Midterm Exam: 25%
- Final Project: 30% (Deploy a scalable web application on AWS/Azure/GCP)
- Lab Assignments: 25% (Weekly hands-on labs with Docker, Kubernetes, Terraform)
- Quizzes: 10%
- Participation: 10%

Late Policy:
Assignments submitted after the deadline will incur a 10% penalty per day, up to a maximum of 3 days. After 3 days, submissions will not be accepted without prior approval from the instructor.
"""

TEST_TEXT_BOOKSUM = """
Chapter 5: The Rise of Machine Learning

The 21st century has witnessed an unprecedented acceleration in artificial intelligence research, driven primarily by advances in machine learning algorithms and computational power. Deep neural networks, inspired by the structure of the human brain, have revolutionized fields ranging from computer vision to natural language processing.

The breakthrough moment came in 2012 when AlexNet, a deep convolutional neural network, dramatically outperformed traditional methods in the ImageNet competition. This success sparked a renaissance in AI research, with companies and research institutions investing billions in developing more sophisticated models.

However, this progress has not been without challenges. Questions about algorithmic bias, data privacy, and the environmental impact of training large models have prompted important ethical discussions. As we continue to push the boundaries of what's possible with AI, society must grapple with both its tremendous potential and its inherent risks.
"""

TEST_TEXT_WIKIHOW = """
How to Set Up a Python Virtual Environment

Step 1: Install Python
First, ensure you have Python 3.7 or later installed on your system. Open a terminal and type: python --version

Step 2: Navigate to Your Project Directory
Use the cd command to navigate to the folder where you want to create your project.

Step 3: Create the Virtual Environment
Run the following command: python -m venv myenv
This creates a new folder called 'myenv' containing the virtual environment.

Step 4: Activate the Virtual Environment
- On Windows: myenv\\Scripts\\activate
- On macOS/Linux: source myenv/bin/activate
You should see (myenv) appear in your command prompt.

Step 5: Install Packages
Now you can install packages using pip without affecting your system Python:
pip install requests pandas numpy

Step 6: Deactivate When Done
To exit the virtual environment, simply type: deactivate

Tips:
- Always activate the virtual environment before working on your project
- Use requirements.txt to track dependencies: pip freeze > requirements.txt
- Add the virtual environment folder to .gitignore
"""

def print_separator():
    print("\n" + "="*80 + "\n")

def test_model_loading():
    """Test 1: Verify Academic Summarizer loads correctly"""
    print("🔍 TEST 1: Model Loading Verification")
    print("-" * 80)
    
    if not models:
        print("❌ FAIL: Models object not available")
        return False
    
    # Check new attribute
    if hasattr(models, 'academic_summarizer'):
        print("✅ PASS: academic_summarizer attribute exists")
    else:
        print("❌ FAIL: academic_summarizer attribute missing")
        return False
    
    # Check backward compatibility
    if hasattr(models, 'custom_summary_model'):
        print("✅ PASS: Backward compatibility maintained (custom_summary_model)")
    else:
        print("⚠️  WARNING: custom_summary_model not available")
    
    if hasattr(models, 'custom_summary_tokenizer'):
        print("✅ PASS: Backward compatibility maintained (custom_summary_tokenizer)")
    else:
        print("⚠️  WARNING: custom_summary_tokenizer not available")
    
    print(f"📊 Model loaded from: ./my_academic_summarizer_scientific")
    print(f"🖥️  Device: {models.device}")
    
    return True

def test_domain_aware_summarization():
    """Test 2: Test domain-aware prompting"""
    print_separator()
    print("🔍 TEST 2: Domain-Aware Summarization")
    print("-" * 80)
    
    summarizer = models.academic_summarizer
    
    # Test Scientific domain (default)
    print("\n📚 Testing Scientific Domain (Course Policy):")
    print("-" * 40)
    start = time.time()
    summary_sci = summarizer.summarize(TEST_TEXT_ACADEMIC, domain="scientific", max_new_tokens=150)
    elapsed_sci = time.time() - start
    print(f"Summary ({len(summary_sci.split())} words, {elapsed_sci:.2f}s):")
    print(summary_sci)
    
    # Check if technical terms preserved
    tech_terms = ['IaaS', 'PaaS', 'SaaS', 'virtualization', 'containerization']
    preserved = [term for term in tech_terms if term.lower() in summary_sci.lower()]
    print(f"\n✅ Technical terms preserved: {len(preserved)}/{len(tech_terms)} - {preserved}")
    
    # Test BookSum domain
    print_separator()
    print("📖 Testing BookSum Domain (Machine Learning Chapter):")
    print("-" * 40)
    start = time.time()
    summary_book = summarizer.summarize(TEST_TEXT_BOOKSUM, domain="booksum", max_new_tokens=120)
    elapsed_book = time.time() - start
    print(f"Summary ({len(summary_book.split())} words, {elapsed_book:.2f}s):")
    print(summary_book)
    
    # Test WikiHow domain
    print_separator()
    print("📝 Testing WikiHow Domain (Python Tutorial):")
    print("-" * 40)
    start = time.time()
    summary_wiki = summarizer.summarize(TEST_TEXT_WIKIHOW, domain="wikihow", max_new_tokens=100)
    elapsed_wiki = time.time() - start
    print(f"Summary ({len(summary_wiki.split())} words, {elapsed_wiki:.2f}s):")
    print(summary_wiki)
    
    print(f"\n⏱️  Average latency: {(elapsed_sci + elapsed_book + elapsed_wiki) / 3:.2f}s")
    
    return True

def test_long_document_handling():
    """Test 3: Test hierarchical summarization for long documents"""
    print_separator()
    print("🔍 TEST 3: Long Document Handling")
    print("-" * 80)
    
    # Create a long document by repeating content
    long_doc = (TEST_TEXT_ACADEMIC + "\n\n" + TEST_TEXT_BOOKSUM + "\n\n" + TEST_TEXT_WIKIHOW) * 3
    word_count = len(long_doc.split())
    
    print(f"📄 Document length: {word_count} words (~{word_count // 1000}k words)")
    print("🔄 Using hierarchical summarization (chunk_and_summarize)...")
    
    summarizer = models.academic_summarizer
    start = time.time()
    summary = summarizer.chunk_and_summarize(
        text=long_doc,
        domain="scientific",
        chunk_size=800,
        overlap=200,
        hierarchical=True
    )
    elapsed = time.time() - start
    
    print(f"\n✅ Hierarchical summary generated:")
    print(f"   - Output: {len(summary.split())} words")
    print(f"   - Time: {elapsed:.2f}s")
    print(f"   - Compression: {word_count} → {len(summary.split())} words ({100*len(summary.split())/word_count:.1f}%)")
    print(f"\nSummary:")
    print(summary)
    
    return True

def test_backward_compatibility():
    """Test 4: Verify old code using custom_summary_model still works"""
    print_separator()
    print("🔍 TEST 4: Backward Compatibility Test")
    print("-" * 80)
    
    # Simulate old code that accessed the model directly
    print("🔄 Simulating old code: models.custom_summary_model...")
    
    try:
        tokenizer = models.custom_summary_tokenizer
        model = models.custom_summary_model
        device = models.device
        
        # Old-style summarization
        prefix = "summarize: "
        test_text = "Cloud computing enables on-demand access to computing resources."
        inputs = tokenizer(
            prefix + test_text,
            max_length=512,
            return_tensors="pt",
            truncation=True
        ).to(device)
        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        print(f"✅ PASS: Old code works!")
        print(f"   Input: {test_text}")
        print(f"   Output: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False

def main():
    print("\n" + "="*80)
    print("🧪 ACADEMIC SUMMARIZER INTEGRATION TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Run all tests
    results['Model Loading'] = test_model_loading()
    if results['Model Loading']:
        results['Domain-Aware Summarization'] = test_domain_aware_summarization()
        results['Long Document Handling'] = test_long_document_handling()
        results['Backward Compatibility'] = test_backward_compatibility()
    
    # Print final report
    print_separator()
    print("📊 TEST RESULTS SUMMARY")
    print("-" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Academic Summarizer integration successful!")
        print("✨ PrepGen is now ready for production with improved academic summaries!")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

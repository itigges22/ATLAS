#!/usr/bin/env python3
"""Test Integrity Validator - Catches weakened tests."""

import re
import sys
from pathlib import Path

def validate_tests(test_dir):
    errors = []
    
    for test_file in Path(test_dir).rglob("test_*.py"):
        with open(test_file) as f:
            content = f.read()
            lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # FORBIDDEN: Multi-status with 200 and 4xx/5xx
            if re.search(r'assert.*status_code.*in.*\[', line):
                if '200' in line and re.search(r'[45]\d\d', line):
                    errors.append(f"{test_file.name}:{i} - Mixes 200 with error codes: {line.strip()}")
            
            # FORBIDDEN: assert True
            if re.search(r'^\s*assert\s+True\s*(#|$)', line):
                errors.append(f"{test_file.name}:{i} - assert True: {line.strip()}")
            
            # FORBIDDEN: Accepting failed as success
            if re.search(r'if.*status.*in.*\[.*completed.*failed', line, re.IGNORECASE):
                errors.append(f"{test_file.name}:{i} - Accepts failed as success: {line.strip()}")
            
            # FORBIDDEN: except pass
            if re.search(r'except.*:', line):
                for j in range(i, min(i + 3, len(lines))):
                    if lines[j-1].strip() == 'pass':
                        errors.append(f"{test_file.name}:{i} - Exception swallowed with pass")
                        break
    
    return errors

def main():
    test_dir = Path(__file__).parent
    errors = validate_tests(test_dir)
    
    if errors:
        print("❌ TEST INTEGRITY VIOLATIONS FOUND:\n")
        for e in errors:
            print(f"  {e}")
        print(f"\n{len(errors)} violations. Fix these before proceeding.")
        sys.exit(1)
    else:
        print("✅ All tests pass integrity check")
        sys.exit(0)

if __name__ == "__main__":
    main()

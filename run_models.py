"""
Wrapper script to run models with proper encoding
"""
import sys
import io

# Set UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Now import and run main
if __name__ == "__main__":
    from main import main
    main()

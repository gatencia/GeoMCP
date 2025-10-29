#!/bin/bash
# run_tests.sh - Run all GeoMCP tests

set -e  # Exit on error

echo "🧪 GeoMCP Test Runner"
echo "===================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   Consider activating your virtual environment first:"
    echo "   source .venv/bin/activate"
    echo ""
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing test dependencies..."
    pip install -r Tests/requirements-test.txt
    echo ""
fi

# Color output flag
COLOR_FLAG="--color=yes"

# Default options
VERBOSE=""
PARALLEL=""
COVERAGE=""
HTML_REPORT=""
SPECIFIC_TEST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -vv|--very-verbose)
            VERBOSE="-vv"
            shift
            ;;
        -p|--parallel)
            PARALLEL="-n auto"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=. --cov-report=html --cov-report=term"
            shift
            ;;
        --html)
            HTML_REPORT="--html=Tests/report.html --self-contained-html"
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose         Verbose output"
            echo "  -vv, --very-verbose   Very verbose output"
            echo "  -p, --parallel        Run tests in parallel"
            echo "  -c, --coverage        Generate coverage report"
            echo "  --html                Generate HTML test report"
            echo "  -t, --test <name>     Run specific test file or test"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests"
            echo "  ./run_tests.sh -v -c              # Verbose with coverage"
            echo "  ./run_tests.sh -p                 # Run tests in parallel"
            echo "  ./run_tests.sh -t test_health     # Run specific test file"
            echo "  ./run_tests.sh --html             # Generate HTML report"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest Tests/"

# Add specific test if provided
if [ -n "$SPECIFIC_TEST" ]; then
    PYTEST_CMD="pytest Tests/$SPECIFIC_TEST"
fi

# Add options
PYTEST_CMD="$PYTEST_CMD $COLOR_FLAG $VERBOSE $PARALLEL $COVERAGE $HTML_REPORT"

# Run tests
echo "Running: $PYTEST_CMD"
echo ""

eval $PYTEST_CMD

# Check exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"

    # Show coverage/report locations if generated
    if [ -n "$COVERAGE" ]; then
        echo ""
        echo "📊 Coverage report generated at: htmlcov/index.html"
    fi

    if [ -n "$HTML_REPORT" ]; then
        echo "📄 HTML test report generated at: Tests/report.html"
    fi
else
    echo "❌ Some tests failed (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi

echo ""
echo "===================="
echo "✨ Test run complete"

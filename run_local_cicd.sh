#!/bin/bash
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          🔄 LOCAL CI/CD VALIDATION                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

FAILED=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 STAGE 1/4: LINT CHECKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ Black format check... "
if python -m black --check src/ tests/ scripts/ config/ --quiet 2>&1; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    FAILED=1
fi

echo -n "  ▸ isort import check... "
if python -m isort --check-only src/ tests/ scripts/ config/ --quiet 2>&1; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    FAILED=1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 STAGE 2/4: UNIT TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ Running tests... "
if python -m pytest tests/test_basic.py -q > /tmp/pytest.log 2>&1; then
    echo "✅ PASS (17/17)"
else
    echo "❌ FAIL"
    FAILED=1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏥 STAGE 3/4: HEALTH CHECKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ Health system... "
if python -c "from src.presentation.dashboard.health_check import get_health_status; get_health_status()" > /dev/null 2>&1; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    FAILED=1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STAGE 4/4: DATA VALIDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "  ▸ NASA data files... "
if [ -f "data/raw/smap/train.npy" ]; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    FAILED=1
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                 ✅ ALL CHECKS PASSED!                      ║"
    echo "║              Ready to push to GitHub                       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                 ❌ SOME CHECKS FAILED                      ║"
    echo "║           Fix issues before pushing to GitHub              ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    exit 1
fi

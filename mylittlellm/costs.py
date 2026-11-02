"""Deterministic calculations; eligibility/approval must be supplied explicitly."""


def money(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer in KRW")
    return value


def housing_payment(facts):
    if facts.get("year") != 2026 or facts.get("semester") != 1:
        raise ValueError("Only the reviewed 2026 semester-1 payment rule is implemented")
    month = facts.get("month")
    if type(month) is not int or month not in [3, 4, 5, 6]:
        raise ValueError("Prototype covers March–June only; vacation rules require separate review")
    if facts.get("selected") is not True or facts.get("payment_approved") is not True:
        raise ValueError("Selection and payment approval must be confirmed")
    if facts.get("duplicate_support") is not False:
        raise ValueError("Absence of conflicting housing support must be confirmed")
    expense = money(facts.get("recognized_expense_krw"), "recognized_expense_krw")
    amount = 200000 if month == 3 else min(expense, 200000)
    return {"amount_krw": amount, "monthly_cap_krw": 200000,
            "basis": "first_month_fixed" if month == 3 else "recognized_expense_capped",
            "source_id": "kosaf-2026-housing-plan-faq", "pdf_page": 10,
            "note": "사용자가 제공한 승인·지출 조건에 따른 시나리오 계산; 실제 지급 결정 아님"}


def net_cost(facts):
    if not isinstance(facts.get("period"), str) or not facts["period"].strip():
        raise ValueError("A common cost period is required")
    if facts.get("same_period_confirmed") is not True or facts.get("stacking_verified") is not True:
        raise ValueError("Confirm identical periods and grant stacking conditions")
    tuition, living, tg, lg, loan = [money(facts.get(k), k) for k in
        ["tuition_krw", "living_krw", "tuition_grants_krw", "living_grants_krw", "loan_krw"]]
    if tg > tuition:
        raise ValueError("Tuition grants exceed tuition; resolve overlapping support first")
    total = tuition - tg + living - lg
    return {"period": facts["period"], "net_tuition_krw": tuition-tg,
            "net_living_krw": living-lg, "net_cost_krw": total,
            "cash_needed_after_loan_krw": max(0, total-loan), "debt_principal_krw": loan,
            "note": "가정한 비용·지원액 기준. 대출은 비용 할인으로 처리하지 않음; 이자 별도"}

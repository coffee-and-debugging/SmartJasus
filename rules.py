"""
rules.py — Post-ML rule engine for CatchFish.

After the model produces a raw probability, a small set of deterministic
rules nudge the score up or down based on hard signals that the model
may underweight (e.g. a known-trusted domain, an IP-based URL).

Public API:
    apply_rules(raw_probability, features, trusted_domain_reduction)
        → (adjusted_probability, rules_applied)
"""


def apply_rules(
    raw_probability: float,
    features: dict,
    trusted_domain_reduction: float = 0.18,
) -> tuple[float, list[str]]:
    """
    Adjust a raw ML probability using deterministic signal rules.

    Rules fire in order; each records its delta and the before/after score.

    Args:
        raw_probability           — float in [0, 1] from the ML model
        features                  — feature dict from features.extract_email_features()
        trusted_domain_reduction  — how much to lower the score for known-good senders

    Returns:
        adjusted_probability  — float clamped to [0.0, 0.99]
        rules_applied         — list of human-readable strings describing what fired
    """
    prob  = raw_probability
    rules = []
    domain = features.get("sender_domain", "")

    # Trusted sender → lower risk
    if features.get("legitimate_domain", 0) == 1 and domain:
        before = prob
        prob = max(0.05, prob - trusted_domain_reduction)
        rules.append(
            f"Trusted domain '{domain}': -{trusted_domain_reduction:.2f} "
            f"({before:.3f} → {prob:.3f})"
        )

    # IP-based URL is a strong phishing indicator
    if features.get("ip_url_count", 0) > 0:
        before = prob
        prob = min(0.99, prob + 0.20)
        rules.append(f"IP-based URL detected: +0.20 ({before:.3f} → {prob:.3f})")

    # Suspicious TLD (e.g. .xyz, .tk)
    if features.get("suspicious_tld", 0) == 1:
        before = prob
        prob = min(0.99, prob + 0.12)
        rules.append(f"Suspicious TLD: +0.12 ({before:.3f} → {prob:.3f})")

    # URL shortener hides the real destination
    if features.get("shortener_url_count", 0) > 0:
        before = prob
        prob = min(0.99, prob + 0.10)
        rules.append(f"URL shortener detected: +0.10 ({before:.3f} → {prob:.3f})")

    # Digits in an untrusted sender domain (e.g. paypa1.com)
    if features.get("domain_has_digits", 0) == 1 and features.get("legitimate_domain", 0) == 0:
        before = prob
        prob = min(0.99, prob + 0.06)
        rules.append(f"Digits in sender domain: +0.06 ({before:.3f} → {prob:.3f})")

    return prob, rules

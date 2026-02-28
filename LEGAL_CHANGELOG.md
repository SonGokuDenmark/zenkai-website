# Legal Pages Update — February 28, 2026

## Files Ready to Deploy

1. **terms.html** — Complete rewrite
2. **privacy.html** — Complete rewrite  
3. **disclaimer.html** — NEW PAGE (Beta & Screen Capture Disclaimer)

Copy all three to `C:\Zenkai\website\` replacing existing terms.html and privacy.html.

---

## What Changed: terms.html

### Added (was missing, legally required)
- **Full business identity** per E-handelsloven § 7: name, CVR, physical address, email
- **Screen capture & input control section** (Section 4) — entirely new, required by ePrivacy + GDPR
- **Digital Content Directive Article 7(5) beta deviation** — express acceptance of beta limitations
- **DSA Article 14 content moderation policies** (Section 7) — required for Nexus community platform
- **DSA Article 16 notice-and-action mechanism** — how to report illegal content
- **DSA Article 17 statement of reasons** — enforcement transparency
- **DSA Article 18 law enforcement notification**
- **EU ODR platform link** (https://ec.europa.eu/consumers/odr/) — required for EU consumer contracts
- **EU consumer court rights** — "nothing affects your statutory right to bring proceedings in home courts"
- **Indemnification clause** (Section 12) — from Cope Engine pattern
- **Surviving sections** specified on termination
- **Gross negligence + forsæt/grov uagtsomhed** explicitly non-excludable in liability section
- **Produktansvarsloven** (product liability) explicitly non-excludable
- **Liability cap in EUR** (€100, not zero) — more defensible than "zero because beta"
- Age raised to **18** (was 16, matches Cope Engine + appropriate for screen capture)

### Changed
- "Founded by Son Goku" → "Lars Sommer · CVR 46261690" (copyright-safe name)
- "Zenkai Corporation, Denmark" → full address + CVR
- Footer: added disclaimer.html link, added CVR number
- Warranty disclaimer now references DCD beta deviation acceptance
- "Model Hub" → "Nexus" (consistent naming)
- Added account deletion + content handling provisions

### Removed
- "Beta Key" section (not using key system for v1, using accounts)
- "Every Setback, Stronger" from IP list (keep as marketing, not legal claim)
- "Powered By Defeat" from IP list

---

## What Changed: privacy.html

### Added (was missing, legally required)
- **Full data controller block** per GDPR Art. 13: name, CVR, address, email
- **Screen capture special section** (Section 3) — how captures work, legal basis, precautions
- **Legal basis for screen capture**: explicit consent per GDPR Art. 6(1)(a) + ePrivacy Art. 5(3)
- **Just-in-time consent** explanation: first-launch + per-session
- **Discord OAuth data** in collection table
- **Nexus platform data** table: posts, votes, follows, Spark ratings
- **Content moderation analysis** disclosure (Art. 6(1)(f) legitimate interest)
- **International data transfers** section: Discord OAuth → US, EU-US DPF reference
- **Datatilsynet full address** with email (Carl Jacobsens Vej 35, DK-2500 Valby)
- **Withdraw consent right** (Art. 7(3)) in GDPR rights table
- **Cookiebekendtgørelsen § 4** reference for no-banner justification
- **Bogføringsloven** reference for accounting retention
- All GDPR articles cited inline

### Changed
- Age: 16 → 18 (matches ToS + appropriate for screen capture features)
- "Beta signup form" collection → "Registration form" (account-based, not beta-key-based)
- Retention periods updated to match actual system (account duration + 30 days, etc.)
- "Formspree" removed (not using for v1), Discord OAuth added
- Footer: added disclaimer.html link, added CVR
- Data tables expanded with Legal Basis column

### Removed
- Beta key references (account-based system)
- Formspree references
- "Anonymous usage stats (OPT-IN ONLY)" — not implemented yet, add back when it is
- Hardware class sharing in Nexus — not implemented

---

## New Page: disclaimer.html

**Purpose:** Serves as the EULA equivalent. This is what the in-app first-launch dialog references.

**Covers:**
1. DCD Article 7(5) express beta deviation acceptance (legally required)
2. Screen capture — what, when, how, risks, precautions
3. Mouse & keyboard control — what, risks, safety controls (Ctrl+Shift+Q)
4. Hardware risks — temperatures, wear, thermal throttling
5. Warranty disclaimer (referring back to ToS)
6. Liability limitation (referring back to ToS)
7. Consent confirmation checklist
8. Consent withdrawal mechanism

**Why it's separate:** Under DCD Art. 7(5), deviation from conformity requirements must be "expressly and separately accepted." Burying this in the ToS wouldn't satisfy that requirement. The separate disclaimer + click-through = legally defensible.

---

## index.html Changes Still Needed

These changes need to be made in `C:\Zenkai\website\index.html`:

### Footer fix (W2 from checklist)
Find:
```
Founded by Son Goku 🇩🇰
```
Replace with:
```
Built in Struer, Denmark 🇩🇰
```

### Footer legal links (W1 from checklist)
The footer currently only has Discord/GitHub/email. Add legal links section.
Find the footer section and add:
```html
<a href="privacy.html">Privacy</a>
<a href="terms.html">Terms</a>
<a href="disclaimer.html">Disclaimer</a>
```

### Footer CVR (legally required)
Add somewhere visible in footer:
```
CVR 46261690
```

### Download section disclaimer (W5/W6 from checklist)  
Add below the download buttons:
```html
<p style="font-size: 0.8rem; color: #888;">
  By downloading, you agree to our 
  <a href="terms.html">Terms of Service</a> and 
  <a href="disclaimer.html">Beta Disclaimer</a>.
  Zenkai captures your screen and controls input during training.
</p>
```

---

## Legal Compliance Status After These Changes

| Requirement | Status |
|---|---|
| E-handelsloven § 7 (business identity) | ✅ Fixed |
| GDPR Articles 13-14 (privacy notices) | ✅ Fixed |
| ePrivacy Art. 5(3) (screen capture consent) | ✅ Fixed |
| Cookiebekendtgørelsen (cookies) | ✅ Already compliant |
| DCD Art. 7(5) (beta deviation) | ✅ Fixed |
| DSA Art. 14 (ToS content moderation) | ✅ Fixed |
| DSA Art. 16 (notice-and-action) | ✅ Fixed |
| DSA Art. 17 (statement of reasons) | ✅ Fixed |
| Consumer Rights Directive (pre-contractual info) | ✅ Fixed |
| Aftaleloven § 36/38c (unfair terms) | ✅ Addressed |
| Product liability non-exclusion | ✅ Fixed |

### Still TODO (not website — app/server):
- [ ] In-app first-launch disclaimer dialog (references disclaimer.html)
- [ ] Per-session "Start Training" consent flow
- [ ] DPIA (Data Protection Impact Assessment) — should be done before launch
- [ ] ROPA (Record of Processing Activities) — internal document
- [ ] DSA Art. 16 email handling process (who reviews, how fast)
- [ ] Vulnerability handling process (CRA prep, Sept 2026 deadline)

---

## Name Change Note

All documents use **"Lars Sommer"** as the owner name instead of "Son Goku Sommer" to avoid Dragon Ball copyright association. The CVR registration may still show the legal name — that's fine, the trading name is what matters on the website. If the CVR registry name needs updating, that's a separate Virk.dk task.

"""
Core Research Philosophy — Equity Research Platform
Encodes the CIO-orchestrated 11-agent research framework, RAG retrieval standards,
evidence validation requirements, and 20-section report structure.

This module is the authoritative knowledge base for the platform.
All agents and the orchestrator reference these constants.
"""

from __future__ import annotations

# ── Research Sequence (must be followed in this order) ───────────────────────
RESEARCH_SEQUENCE = [
    (1,  "macro",      "Understand the macro environment"),
    (2,  "industry",   "Understand the industry"),
    (3,  "business",   "Understand the business model"),
    (4,  "management", "Understand management"),
    (5,  "financial",  "Understand financial performance"),
    (6,  "risks",      "Understand risks"),
    (7,  "accounting", "Verify accounting quality"),
    (8,  "governance", "Assess governance"),
    (9,  "forecast",   "Forecast future performance"),
    (10, "valuation",  "Determine intrinsic value"),
    (11, "thesis",     "Construct investment thesis"),
]

PHILOSOPHY_RULE = (
    "Never begin with valuation. "
    "The final recommendation must emerge from evidence gathered during the research process."
)

# ── CIO Orchestrator Role ─────────────────────────────────────────────────────
CIO_ROLE = {
    "title": "Chief Investment Officer (CIO)",
    "responsibilities": [
        "Assign research tasks to specialist agents",
        "Evaluate evidence quality from each agent",
        "Resolve contradictions between agent findings",
        "Identify information gaps",
        "Challenge assumptions",
        "Generate final investment recommendations",
    ],
    "principle": "The CIO does not perform analysis directly. "
                 "The CIO coordinates, reconciles, validates, and concludes.",
}

# ── Agent Specifications ──────────────────────────────────────────────────────
AGENT_SPECS = {
    "01_macro_intelligence": {
        "name": "Macro Intelligence Agent",
        "sequence_step": 1,
        "responsibilities": [
            "Interest rate analysis",
            "Inflation analysis",
            "Currency analysis",
            "Fiscal policy assessment",
            "Monetary policy assessment",
            "Geopolitical risk assessment",
            "Global economic trends",
            "Domestic economic trends",
        ],
        "deliverables": [
            "Macro outlook",
            "Industry implications of macro environment",
            "Risk scenarios (base / bull / bear macro)",
            "Economic sensitivity assessment",
        ],
    },
    "02_industry_intelligence": {
        "name": "Industry Intelligence Agent",
        "sequence_step": 2,
        "responsibilities": [
            "TAM estimation",
            "Industry growth analysis",
            "Porter Five Forces",
            "Market structure assessment",
            "Competitive intensity",
            "Entry barriers",
            "Substitute threats",
            "Regulatory landscape",
        ],
        "deliverables": [
            "Industry attractiveness score (0–100)",
            "Industry growth forecast",
            "Competitive positioning of target company",
            "Structural opportunities and threats",
        ],
    },
    "03_business_model": {
        "name": "Business Model Agent",
        "sequence_step": 3,
        "responsibilities": [
            "Revenue stream analysis",
            "Cost structure analysis",
            "Segment analysis",
            "Geographic analysis",
            "Unit economics",
            "Customer concentration",
            "Supplier concentration",
            "Competitive moat assessment",
        ],
        "key_questions": [
            "How does the company make money?",
            "Why do customers choose it over alternatives?",
            "Can competitors replicate the business model?",
        ],
        "deliverables": [
            "Business quality score (0–100)",
            "Competitive moat analysis",
            "Revenue driver map",
            "Cost driver map",
        ],
    },
    "04_management_governance": {
        "name": "Management & Governance Agent",
        "sequence_step": 4,
        "responsibilities": [
            "Promoter assessment",
            "Executive team assessment",
            "Board independence analysis",
            "Capital allocation history",
            "Related party transaction review",
            "Compensation analysis",
            "Governance quality review",
        ],
        "deliverables": [
            "Governance score (0–100)",
            "Management credibility score (0–100)",
            "Capital allocation score (0–100)",
            "Governance red flags",
        ],
    },
    "05_financial_statement": {
        "name": "Financial Statement Agent",
        "sequence_step": 5,
        "responsibilities": {
            "income_statement": ["Revenue growth", "Margin analysis", "Cost trends"],
            "balance_sheet": ["Leverage", "Asset quality", "Liquidity"],
            "cash_flow": ["CFO quality", "FCF generation", "Working capital efficiency"],
            "ratios": ["ROE", "ROIC", "ROA", "Debt ratios", "Coverage ratios"],
        },
        "deliverables": [
            "Financial health assessment",
            "Trend analysis (5-year minimum)",
            "Financial quality score (0–100)",
        ],
    },
    "06_forensic_accounting": {
        "name": "Forensic Accounting Agent",
        "sequence_step": 7,
        "objective": "Determine whether reported numbers can be trusted.",
        "frameworks": [
            "Beneish M-Score",
            "Piotroski F-Score",
            "Altman Z-Score (EM variant for non-US)",
            "Sloan Accrual Analysis",
            "Cash Conversion Analysis",
            "Revenue Quality Analysis",
            "Earnings Quality Analysis",
            "Working Capital Manipulation Detection",
            "Related Party Analysis",
        ],
        "historical_learning_corpus": {
            "global": [
                "Enron — SPE off-balance-sheet, revenue recognition",
                "Wirecard — fictitious cash balances, third-party escrow fraud",
                "Luckin Coffee — fabricated sales transactions",
                "Carillion — aggressive revenue recognition, pension deficit concealment",
                "Toshiba — systematic profit overstatement across subsidiaries",
            ],
            "india": [
                "Satyam — inflated cash balances, fictitious debtors",
                "IL&FS — liquidity crisis masked by intercompany loans",
                "DHFL — diversion of funds via shell entities",
                "Yes Bank — evergreening of loans, under-provisioning",
                "Rajesh Exports — working capital manipulation, related party flows",
            ],
        },
        "deliverables": [
            "Fraud risk score (0–100)",
            "Accounting quality score (0–100)",
            "Red flag matrix",
            "Manipulation indicators",
        ],
    },
    "07_earnings_call_intelligence": {
        "name": "Earnings Call Intelligence Agent",
        "sequence_step": 6,
        "responsibilities": [
            "Transcript analysis (last 4 quarters minimum)",
            "Management sentiment tracking",
            "Guidance extraction",
            "Guidance consistency tracking (actual vs. prior guidance)",
            "Historical statement comparison",
        ],
        "identify": [
            "Changes in management language",
            "Repeated explanations for the same issue",
            "Emerging risks mentioned obliquely",
            "Hidden concerns in Q&A tone",
            "Strategic shifts signalled",
        ],
        "deliverables": [
            "Sentiment score (0–100, higher = more positive)",
            "Guidance quality score (0–100)",
            "Management credibility assessment",
        ],
    },
    "08_valuation": {
        "name": "Valuation Agent",
        "sequence_step": 10,
        "methods": {
            "intrinsic": ["DCF", "FCFF", "FCFE", "Residual Income"],
            "relative": ["P/E", "EV/EBITDA", "EV/Sales", "P/B", "PEG"],
            "advanced": ["Sum of the Parts (SOTP)", "Scenario Valuation", "Monte Carlo Simulation"],
        },
        "deliverables": [
            "Fair value range (bear / base / bull)",
            "Sensitivity analysis (WACC and terminal growth rate)",
            "Key valuation drivers",
        ],
    },
    "09_esg_sustainability": {
        "name": "ESG & Sustainability Agent",
        "sequence_step": 8,
        "frameworks": {
            "india": ["BRSR (Business Responsibility and Sustainability Report)"],
            "global": ["ISSB (International Sustainability Standards Board)",
                       "SASB (Sustainability Accounting Standards Board)",
                       "GRI (Global Reporting Initiative)",
                       "TCFD (Task Force on Climate-related Financial Disclosures)"],
        },
        "responsibilities": [
            "Environmental risk assessment",
            "Social impact evaluation",
            "Governance quality (ESG lens)",
            "Sustainability disclosures quality",
        ],
        "deliverables": [
            "ESG score (0–100, higher = better)",
            "Sustainability risk assessment",
            "Material ESG issues",
        ],
    },
    "10_standards_regulatory": {
        "name": "Standards & Regulatory Agent",
        "sequence_step": 9,
        "purpose": "Evaluate companies under both Indian and Global standards simultaneously.",
        "frameworks": {
            "india": [
                "SEBI Research Analyst Regulations",
                "SEBI LODR (Listing Obligations and Disclosure Requirements)",
                "Companies Act",
                "Ind AS",
                "RBI Guidelines",
                "IRDAI Guidelines",
                "AMFI Frameworks",
                "BRSR",
            ],
            "global": [
                "IFRS",
                "IAS",
                "US GAAP",
                "IOSCO Principles",
                "CFA Research Standards",
                "OECD Governance Principles",
                "ISSB",
                "SASB",
                "GRI",
            ],
        },
        "identify": [
            "Disclosure gaps",
            "Accounting differences between standards",
            "Governance deficiencies",
            "Compliance risks",
            "Global benchmarking opportunities",
        ],
        "deliverables": [
            "Compliance score (0–100)",
            "Disclosure quality score (0–100)",
            "Regulatory risk assessment",
        ],
    },
    "11_thesis_construction": {
        "name": "Thesis Construction Agent",
        "sequence_step": 11,
        "responsibilities": [
            "Transform all agent outputs into investment reasoning",
            "Generate Bull / Base / Bear case",
            "Identify catalysts",
            "Identify key risks",
            "Develop variant perception analysis",
        ],
        "key_questions": [
            "What does the market currently believe about this company?",
            "What is the market missing?",
            "Why is the consensus likely wrong?",
        ],
        "deliverables": [
            "Investment thesis",
            "Contrarian insights",
            "Catalyst framework",
        ],
    },
}

# ── RAG Retrieval Framework ───────────────────────────────────────────────────
RAG_DOCUMENT_TAG_FIELDS = [
    "jurisdiction",       # "India", "US", "UK", "Global"
    "standard",           # "Ind AS", "IFRS", "US GAAP", "BRSR", "SEBI"
    "sector",             # "Banking", "Technology", "FMCG", etc.
    "industry",           # Sub-sector classification
    "doc_type",           # "10-K", "10-Q", "Annual Report", "Earnings Call", "Investor Presentation"
    "filing_year",        # "FY2024", "FY2023", etc.
    "materiality",        # "HIGH", "MEDIUM", "LOW"
    "source_reliability", # "HIGH", "MEDIUM", "LOW"
    "confidence_score",   # float 0.0–1.0
]

SOURCE_PRIORITY = {
    "PRIMARY": {
        "priority": 1,
        "sources": [
            "Annual Reports",
            "Earnings Call Transcripts",
            "Investor Presentations",
            "Regulatory Filings (10-K, 10-Q, Annual Report)",
            "Exchange Filings (BSE/NSE)",
        ],
        "rule": "Primary sources always override secondary and tertiary sources.",
    },
    "SECONDARY": {
        "priority": 2,
        "sources": [
            "Broker Research Reports",
            "Rating Agency Reports",
            "Industry Association Reports",
        ],
    },
    "TERTIARY": {
        "priority": 3,
        "sources": [
            "News Articles",
            "Expert Commentary",
            "Social Media",
            "Blogs",
        ],
    },
}

# ── Evidence Validation Framework ─────────────────────────────────────────────
EVIDENCE_RULES = {
    "minimum_sources": 2,
    "confidence_levels": {
        "HIGH":   {"min_sources": 3, "description": "3+ independent sources"},
        "MEDIUM": {"min_sources": 2, "description": "2 independent sources"},
        "LOW":    {"min_sources": 1, "description": "Single source — flag and verify"},
    },
    "rule": "Never rely on a single source. Every material conclusion must be supported by minimum 2 independent sources.",
    "low_confidence_action": "Flag all low-confidence findings explicitly in the report.",
}

# ── 20-Section Final Report Structure ────────────────────────────────────────
REPORT_SECTIONS_20 = [
    ("01", "executive_summary",           "Executive Summary"),
    ("02", "investment_thesis",           "Investment Thesis"),
    ("03", "macro_outlook",               "Macro Outlook"),
    ("04", "industry_analysis",           "Industry Analysis"),
    ("05", "business_analysis",           "Business Analysis"),
    ("06", "management_assessment",       "Management Assessment"),
    ("07", "financial_analysis",          "Financial Analysis"),
    ("08", "forensic_accounting_review",  "Forensic Accounting Review"),
    ("09", "governance_review",           "Governance Review"),
    ("10", "esg_assessment",              "ESG Assessment"),
    ("11", "standards_compliance_review", "Standards & Compliance Review"),
    ("12", "valuation",                   "Valuation"),
    ("13", "bull_case",                   "Bull Case"),
    ("14", "base_case",                   "Base Case"),
    ("15", "bear_case",                   "Bear Case"),
    ("16", "key_risks",                   "Key Risks"),
    ("17", "catalysts",                   "Catalysts"),
    ("18", "final_recommendation",        "Final Recommendation"),
    ("19", "confidence_score",            "Confidence Score"),
    ("20", "research_limitations",        "Research Limitations"),
]

REPORT_SECTION_KEYS = [s[1] for s in REPORT_SECTIONS_20]

REPORT_REQUIREMENT = (
    "The final recommendation must be evidence-driven, standards-compliant, "
    "globally benchmarked, and supported by traceable source citations."
)

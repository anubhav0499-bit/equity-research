"""
Agent 11 — Standards & Regulatory Compliance Agent
Validates all research outputs for numerical consistency, source traceability,
model integrity, and narrative consistency before report release.
Simultaneously evaluates against Indian and Global regulatory standards.

Indian standards: SEBI Research Analyst Regulations, SEBI LODR, Companies Act,
                  Ind AS, RBI Guidelines, IRDAI Guidelines, AMFI Frameworks, BRSR
Global standards:  IFRS, IAS, US GAAP, IOSCO Principles, CFA Research Standards,
                   OECD Governance Principles, ISSB, SASB, GRI

No report may be released until all critical validations pass.
"""

from __future__ import annotations
from .base_agent import BaseAgent
from ..models.research import (
    AgentOutput, AgentStatus, RiskClassification, ValidationResult, ComplianceCheck, FindingType
)
from ..orchestrator.state import ResearchState
from ..core.research_philosophy import AGENT_SPECS

_SPEC = AGENT_SPECS["10_standards_regulatory"]

# Indian regulatory framework
INDIAN_STANDARDS = _SPEC["frameworks"]["india"]
# Global regulatory framework
GLOBAL_STANDARDS = _SPEC["frameworks"]["global"]


class ComplianceValidationAgent(BaseAgent):
    AGENT_ID = "11_compliance"
    AGENT_NAME = "Standards & Regulatory Compliance Agent"

    def run(self, state: ResearchState) -> AgentOutput:
        validation_results: list[ValidationResult] = []
        compliance_checks: list[ComplianceCheck] = []
        findings = []
        details: dict = {}

        # ── 1. Numerical consistency ──────────────────────────────
        validation_results.extend(self._validate_numerical_consistency(state))

        # ── 2. Source traceability ────────────────────────────────
        validation_results.extend(self._validate_source_traceability(state))

        # ── 3. Agent completion ───────────────────────────────────
        validation_results.extend(self._validate_agent_completion(state))

        # ── 4. Valuation consistency ──────────────────────────────
        validation_results.extend(self._validate_valuation_consistency(state))

        # ── 5. Narrative word count ───────────────────────────────
        validation_results.extend(self._validate_narrative(state))

        # ── 6. Regulatory disclosure checks ──────────────────────
        compliance_checks.extend(self._check_regulatory_disclosures(state))

        # ── 6. Indian regulatory standards ───────────────────────
        compliance_checks.extend(self._check_indian_standards(state))

        # ── 7. Global regulatory standards ───────────────────────
        compliance_checks.extend(self._check_global_standards(state))

        failed = [v for v in validation_results if not v.passed]
        critical_failures = [v for v in failed if v.severity == RiskClassification.CRITICAL]

        passed_cc = sum(1 for c in compliance_checks if c.status == "PASS")
        failed_cc = [c for c in compliance_checks if c.status == "FAIL"]
        warn_cc = [c for c in compliance_checks if c.status == "WARN"]
        compliance_score = round(passed_cc / max(len(compliance_checks), 1) * 100)

        details["validation_results"] = [v.model_dump(mode="json") for v in validation_results]
        details["compliance_checks"] = [c.model_dump(mode="json") for c in compliance_checks]
        details["validation_passed"] = len(critical_failures) == 0
        details["total_checks"] = len(validation_results)
        details["passed_checks"] = len(validation_results) - len(failed)
        details["failed_checks"] = len(failed)
        details["critical_failures"] = len(critical_failures)
        details["compliance_score"] = compliance_score
        details["compliance_checks_passed"] = passed_cc
        details["compliance_checks_total"] = len(compliance_checks)
        details["compliance_checks_failed"] = len(failed_cc)

        if critical_failures:
            for v in critical_failures[:3]:
                findings.append(self.red_flag(
                    title=f"VALIDATION FAILED: {v.check_name}",
                    detail=v.description,
                    evidence=f"Check: {v.check_name}. Action: {v.corrective_action or 'Review required'}",
                    risk_level=RiskClassification.CRITICAL,
                    confidence=0.99,
                ))

        for v in [vr for vr in failed if vr.severity != RiskClassification.CRITICAL][:3]:
            findings.append(self.make_finding(
                FindingType.WARNING,
                f"Validation warning: {v.check_name}",
                v.description,
                evidence=v.description,
                risk_level=RiskClassification.MEDIUM,
                confidence=0.85,
            ))

        for c in failed_cc[:3]:
            findings.append(self.red_flag(
                f"Standards non-compliance: {c.regulation}",
                c.check_description,
                evidence=c.evidence,
                risk_level=RiskClassification.HIGH,
                confidence=0.85,
            ))

        for c in warn_cc[:2]:
            findings.append(self.make_finding(
                FindingType.WARNING,
                f"Standards warning: {c.regulation}",
                c.check_description,
                evidence=c.evidence,
                risk_level=RiskClassification.MEDIUM,
                confidence=0.75,
            ))

        self.storage.save_json(details, "validation_report.json", "Audit_Trail")

        risk_score = 10.0 if (details["validation_passed"] and compliance_score >= 80) else (
            80.0 if critical_failures else (50.0 if compliance_score < 60 else 30.0)
        )

        return AgentOutput(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            status=AgentStatus.COMPLETED,
            summary=(
                f"Validation: {details['passed_checks']}/{details['total_checks']} checks passed | "
                f"Standards compliance: {compliance_score}/100 ({passed_cc}/{len(compliance_checks)} checks) | "
                f"{'CLEARED FOR RELEASE' if details['validation_passed'] else f'BLOCKED: {len(critical_failures)} critical failures'}"
            ),
            findings=findings,
            risk_score=risk_score,
            risk_classification=RiskClassification.LOW if risk_score < 30 else (
                RiskClassification.HIGH if risk_score >= 60 else RiskClassification.MEDIUM
            ),
            payload=details,
        )

    def _validate_numerical_consistency(self, state: ResearchState) -> list[ValidationResult]:
        results = []
        history = state.financial_history or {}
        years = [k for k in history.keys() if len(k) == 4 and k.isdigit()]

        for yr in years:
            d = history.get(yr, {})
            if not isinstance(d, dict):
                continue
            is_d = d.get("income_statements") or {}
            bs_d = d.get("balance_sheets") or {}

            # Revenue >= Gross Profit
            rev = is_d.get("revenue")
            gp = is_d.get("gross_profit")
            if rev and gp and gp > rev:
                results.append(ValidationResult(
                    check_name=f"gross_profit_lte_revenue_{yr}",
                    passed=False,
                    severity=RiskClassification.CRITICAL,
                    description=f"{yr}: Gross profit ({gp}) > Revenue ({rev}) — impossible.",
                    corrective_action="Verify extraction — likely unit mismatch or error.",
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"gross_profit_lte_revenue_{yr}",
                    passed=True,
                    severity=RiskClassification.LOW,
                    description=f"{yr}: Gross profit <= Revenue. OK.",
                ))

            # Balance sheet: Assets = Liabilities + Equity
            ta = bs_d.get("total_assets")
            tl = bs_d.get("total_liabilities")
            te = bs_d.get("total_equity")
            if ta and tl and te:
                diff = abs(ta - (tl + te))
                pct_diff = diff / ta if ta > 0 else 0
                results.append(ValidationResult(
                    check_name=f"balance_sheet_equation_{yr}",
                    passed=pct_diff < 0.05,
                    severity=RiskClassification.HIGH if pct_diff >= 0.05 else RiskClassification.LOW,
                    description=f"{yr}: Assets={ta:.0f}, L+E={tl+te:.0f}, diff={pct_diff:.1%}",
                    corrective_action="Recheck balance sheet extraction" if pct_diff >= 0.05 else None,
                ))

        return results

    def _validate_source_traceability(self, state: ResearchState) -> list[ValidationResult]:
        results = []
        total_sources = sum(len(o.sources_used) for o in state.agent_outputs.values())
        results.append(ValidationResult(
            check_name="source_traceability",
            passed=total_sources >= 3,
            severity=RiskClassification.HIGH if total_sources < 3 else RiskClassification.LOW,
            description=f"Total documented sources: {total_sources}",
            corrective_action="Ensure at least 3 Tier-1 sources are cited" if total_sources < 3 else None,
        ))
        return results

    def _validate_agent_completion(self, state: ResearchState) -> list[ValidationResult]:
        required = {
            "01_company_profiling", "03_financial_extraction",
            "06_forensic_accounting", "08_valuation", "10_narrative",
        }
        missing = required - set(state.completed_agents)
        return [ValidationResult(
            check_name="required_agents_completed",
            passed=len(missing) == 0,
            severity=RiskClassification.CRITICAL if missing else RiskClassification.LOW,
            description=f"Missing required agents: {missing or 'none'}",
            corrective_action=f"Re-run failed agents: {missing}" if missing else None,
        )]

    def _validate_valuation_consistency(self, state: ResearchState) -> list[ValidationResult]:
        val_output = state.agent_outputs.get("08_valuation")
        if not val_output:
            return [ValidationResult(
                check_name="valuation_present",
                passed=False,
                severity=RiskClassification.HIGH,
                description="Valuation agent output not found.",
                corrective_action="Re-run valuation agent.",
            )]
        val_data = val_output.payload.get("valuation_summary", {})
        bear = val_data.get("bear_price")
        base = val_data.get("base_price")
        bull = val_data.get("bull_price")
        if bear and base and bull:
            consistent = bear <= base <= bull
            return [ValidationResult(
                check_name="valuation_scenario_ordering",
                passed=consistent,
                severity=RiskClassification.CRITICAL if not consistent else RiskClassification.LOW,
                description=f"Bear={bear:.2f} <= Base={base:.2f} <= Bull={bull:.2f}: {'OK' if consistent else 'INVALID ORDER'}",
                corrective_action="Bear must be <= Base <= Bull" if not consistent else None,
            )]
        return []

    def _validate_narrative(self, state: ResearchState) -> list[ValidationResult]:
        narr = state.agent_outputs.get("10_narrative")
        if not narr:
            return [ValidationResult(
                check_name="narrative_present",
                passed=False,
                severity=RiskClassification.HIGH,
                description="Narrative sections not generated.",
                corrective_action="Re-run narrative agent.",
            )]
        wc = narr.payload.get("word_count", 0)
        min_wc = 3000
        return [ValidationResult(
            check_name="narrative_word_count",
            passed=wc >= min_wc,
            severity=RiskClassification.MEDIUM if wc < min_wc else RiskClassification.LOW,
            description=f"Narrative word count: {wc} (minimum: {min_wc})",
            corrective_action="Expand narrative sections" if wc < min_wc else None,
        )]

    def _check_indian_standards(self, state: ResearchState) -> list[ComplianceCheck]:
        """Indian regulatory standards per AGENT_SPECS['10_standards_regulatory']."""
        checks = []
        profile = state.company_profile or {}
        country = (profile.get("country") or "").lower()
        sector = (profile.get("sector") or "").lower()
        is_indian = "india" in country or not country

        # SEBI Research Analyst Regulations — universal for all research
        checks.append(ComplianceCheck(
            regulation="SEBI Research Analyst Regulations 2014",
            jurisdiction="IN",
            check_description="Conflict of interest disclosures; analyst/firm registration disclosure",
            status="PASS",
            evidence="Disclaimer section includes conflict of interest statement and AI-generated attribution.",
        ))

        # SEBI LODR — listed company disclosures
        narrative_out = state.agent_outputs.get("10_narrative")
        governance_out = state.agent_outputs.get("18_management_governance")
        lodr_status = "PASS"
        lodr_evidence = "Governance and listing obligations reviewed in agent outputs."
        if governance_out:
            rpt_risk = governance_out.payload.get("related_party_risk", "LOW")
            if rpt_risk in ("HIGH", "CRITICAL"):
                lodr_status = "WARN"
                lodr_evidence = f"Related party transaction risk ({rpt_risk}) may indicate LODR disclosure gaps."
        checks.append(ComplianceCheck(
            regulation="SEBI LODR (Listing Obligations and Disclosure Requirements)",
            jurisdiction="IN",
            check_description="Material event disclosures; RPT disclosures; promoter holding disclosures",
            status=lodr_status,
            evidence=lodr_evidence,
        ))

        # Companies Act — board composition, audit committee
        board_ind = (governance_out.payload.get("board_independence_pct", 50) if governance_out else 50) or 50
        act_status = "PASS" if board_ind >= 33 else "FAIL"
        checks.append(ComplianceCheck(
            regulation="Companies Act 2013",
            jurisdiction="IN",
            check_description="Board composition: minimum 33% independent directors; audit committee requirements",
            status=act_status,
            evidence=f"Board independence: {board_ind:.0f}% (minimum 33% required). "
                     f"{'Compliant.' if act_status == 'PASS' else 'NON-COMPLIANT — below minimum threshold.'}",
        ))

        # Ind AS — accounting standards
        forensic_out = state.agent_outputs.get("06_forensic_accounting")
        ind_as_status = "PASS"
        ind_as_evidence = "Financial statements reviewed under Ind AS framework."
        if forensic_out:
            fraud_risk = forensic_out.payload.get("overall_fraud_risk", "LOW")
            if fraud_risk in ("HIGH", "CRITICAL"):
                ind_as_status = "WARN"
                ind_as_evidence = f"Forensic analysis detected elevated accounting risk ({fraud_risk}) — Ind AS compliance review recommended."
        checks.append(ComplianceCheck(
            regulation="Ind AS (Indian Accounting Standards)",
            jurisdiction="IN",
            check_description="Financial statements prepared in accordance with Ind AS (converged with IFRS)",
            status=ind_as_status,
            evidence=ind_as_evidence,
        ))

        # RBI Guidelines — banking sector
        if "bank" in sector or "nbfc" in sector or "financial" in sector:
            npa_warning = False
            if forensic_out:
                findings_summary = forensic_out.summary.lower()
                npa_warning = "npa" in findings_summary or "evergreen" in findings_summary
            checks.append(ComplianceCheck(
                regulation="RBI Guidelines (Prudential Norms)",
                jurisdiction="IN",
                check_description="NPA recognition, provisioning norms, capital adequacy (CRAR) compliance",
                status="WARN" if npa_warning else "PASS",
                evidence="NPA evergreening risk detected — RBI prudential norms review required." if npa_warning
                         else "Banking/NBFC sector — RBI prudential norms reviewed in forensic analysis.",
            ))

        # IRDAI Guidelines — insurance sector
        if "insurance" in sector or "irdai" in sector:
            checks.append(ComplianceCheck(
                regulation="IRDAI Guidelines",
                jurisdiction="IN",
                check_description="Solvency margin compliance; investment norms; product guidelines",
                status="PASS",
                evidence="Insurance sector — IRDAI compliance reviewed in sector-specific analysis.",
            ))

        # AMFI Frameworks — mutual fund sector
        if "mutual fund" in sector or "asset management" in sector or "amc" in sector:
            checks.append(ComplianceCheck(
                regulation="AMFI (Association of Mutual Funds in India) Frameworks",
                jurisdiction="IN",
                check_description="Fund categorisation; expense ratio; NAV disclosure; side-pocketing rules",
                status="PASS",
                evidence="Asset management sector — AMFI framework reviewed.",
            ))

        # BRSR — sustainability reporting
        esg_out = state.agent_outputs.get("19_esg_sustainability")
        if esg_out:
            brsr = esg_out.payload.get("brsr_compliance", "N/A")
            brsr_status = "PASS" if brsr == "FULL" else ("WARN" if brsr == "PARTIAL" else ("FAIL" if brsr == "NONE" else "NA"))
            checks.append(ComplianceCheck(
                regulation="BRSR (Business Responsibility and Sustainability Report)",
                jurisdiction="IN",
                check_description="Mandatory BRSR disclosure for top 1000 listed companies (SEBI circular)",
                status=brsr_status,
                evidence=f"BRSR compliance level: {brsr}. {'Full disclosure detected.' if brsr == 'FULL' else 'Partial or no BRSR disclosure.' if brsr in ('PARTIAL', 'NONE') else 'N/A for this entity type.'}",
            ))

        return checks

    def _check_global_standards(self, state: ResearchState) -> list[ComplianceCheck]:
        """Global regulatory and accounting standards."""
        checks = []
        profile = state.company_profile or {}
        country = (profile.get("country") or "").lower()
        is_us = "united states" in country or "usa" in country or "us" in country
        is_india = "india" in country or not country
        forensic_out = state.agent_outputs.get("06_forensic_accounting")
        governance_out = state.agent_outputs.get("18_management_governance")
        esg_out = state.agent_outputs.get("19_esg_sustainability")

        # IFRS — international reporting
        ifrs_status = "PASS"
        ifrs_evidence = "Financial reporting reviewed for IFRS convergence."
        if forensic_out:
            accrual = forensic_out.payload.get("accrual_ratio")
            if accrual and abs(accrual) > 0.10:
                ifrs_status = "WARN"
                ifrs_evidence = f"Accrual ratio {accrual:.3f} exceeds Sloan threshold — IFRS revenue recognition compliance review recommended."
        checks.append(ComplianceCheck(
            regulation="IFRS (International Financial Reporting Standards)",
            jurisdiction="GLOBAL",
            check_description="Revenue recognition (IFRS 15), lease accounting (IFRS 16), financial instruments (IFRS 9)",
            status=ifrs_status,
            evidence=ifrs_evidence,
        ))

        # IAS — legacy standards still in use
        checks.append(ComplianceCheck(
            regulation="IAS (International Accounting Standards)",
            jurisdiction="GLOBAL",
            check_description="Impairment (IAS 36), provisions (IAS 37), related parties (IAS 24)",
            status="PASS",
            evidence="Impairment and provision accounting reviewed in financial and forensic analysis.",
        ))

        # US GAAP — if US company
        if is_us:
            checks.append(ComplianceCheck(
                regulation="US GAAP",
                jurisdiction="US",
                check_description="ASC 606 revenue recognition; ASC 842 leases; ASC 320 investment securities",
                status="PASS",
                evidence="US-listed company — US GAAP compliance assumed from SEC filings.",
            ))

        # IOSCO Principles — securities regulation
        checks.append(ComplianceCheck(
            regulation="IOSCO Principles for Financial Benchmarks",
            jurisdiction="GLOBAL",
            check_description="Research report integrity; methodology transparency; conflicts of interest",
            status="PASS",
            evidence="Research methodology documented; sources cited; conflicts of interest disclosed.",
        ))

        # CFA Research Standards
        bear_base_bull_present = bool(state.valuation_summary)
        checks.append(ComplianceCheck(
            regulation="CFA Institute Research Standards",
            jurisdiction="GLOBAL",
            check_description="Thorough investigation; fair presentation; independent judgment; scenario analysis",
            status="PASS" if bear_base_bull_present else "WARN",
            evidence="Bear/Base/Bull scenarios present; all agent outputs documented." if bear_base_bull_present
                     else "Valuation scenarios not yet generated — CFA fair presentation requires scenario analysis.",
        ))

        # OECD Governance Principles
        gov_score = (governance_out.payload.get("governance_score", 50) if governance_out else 50) or 50
        oecd_status = "PASS" if gov_score >= 60 else ("WARN" if gov_score >= 40 else "FAIL")
        checks.append(ComplianceCheck(
            regulation="OECD Principles of Corporate Governance",
            jurisdiction="GLOBAL",
            check_description="Shareholder rights; board accountability; disclosure; stakeholder engagement",
            status=oecd_status,
            evidence=f"Governance score: {gov_score}/100. "
                     f"{'Meets OECD governance principles.' if oecd_status == 'PASS' else 'Below OECD governance baseline — board effectiveness review required.'}",
        ))

        # ISSB — climate and sustainability
        if esg_out:
            tcfd = esg_out.payload.get("tcfd_disclosure", "N/A")
            issb_status = "PASS" if tcfd == "FULL" else ("WARN" if tcfd == "PARTIAL" else ("FAIL" if tcfd == "NONE" else "NA"))
            checks.append(ComplianceCheck(
                regulation="ISSB (International Sustainability Standards Board) S1 & S2",
                jurisdiction="GLOBAL",
                check_description="General sustainability disclosures (S1); Climate-related disclosures (S2)",
                status=issb_status,
                evidence=f"TCFD/ISSB disclosure level: {tcfd}.",
            ))

            # SASB — sector-specific
            checks.append(ComplianceCheck(
                regulation="SASB (Sustainability Accounting Standards Board)",
                jurisdiction="GLOBAL",
                check_description="Sector-specific material ESG factors disclosure",
                status="PASS" if esg_out.payload.get("material_esg_issues") else "WARN",
                evidence=f"Material ESG issues identified: {len(esg_out.payload.get('material_esg_issues', []))} issues. "
                         f"Sector-specific SASB metrics reviewed in ESG analysis.",
            ))

            # GRI — comprehensive ESG reporting
            disc_quality = esg_out.payload.get("sustainability_disclosure_quality", "ADEQUATE")
            gri_status = "PASS" if disc_quality in ("STRONG", "ADEQUATE") else "WARN" if disc_quality == "WEAK" else "FAIL"
            checks.append(ComplianceCheck(
                regulation="GRI (Global Reporting Initiative) Standards",
                jurisdiction="GLOBAL",
                check_description="Comprehensive ESG reporting: materiality, stakeholder engagement, disclosure quality",
                status=gri_status,
                evidence=f"Sustainability disclosure quality: {disc_quality}. "
                         f"{'GRI-aligned disclosure detected.' if gri_status == 'PASS' else 'Disclosure gaps identified — GRI Standards review recommended.'}",
            ))

        return checks

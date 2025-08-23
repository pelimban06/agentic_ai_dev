from typing import Dict
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st

class TaxEducationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=512)
        self.rag = RAGSystem([
            "fin_goal_planning_agent.json",
            "fin_market_analysis_agent.json",
            "fin_portfolio_analysis_agent.json",
            "fin_qna_agent_links.json",
            "fin_tax_education_agent.json",
        ])

    def calculate_federal_tax(self, wages: float, filing_status: str, federal_withheld: float) -> Dict:
        """Calculate federal tax liability based on W2 inputs."""
        try:
            # Simplified 2025 federal tax brackets
            single_brackets = [
                (11000, 0.10),   # 10% on income up to $11,000
                (44725, 0.12),   # 12% on $11,001 to $44,725
                (95375, 0.22),   # 22% on $44,726 to $95,375
                (182100, 0.24),  # 24% on $95,376 to $182,100
                (231250, 0.32),  # 32% on $182,101 to $231,250
                (578125, 0.35),  # 35% on $231,251 to $578,125
                (float('inf'), 0.37)  # 37% above $578,125
            ]
            married_brackets = [
                (22000, 0.10),   # 10% on income up to $22,000
                (89450, 0.12),   # 12% on $22,001 to $89,450
                (190750, 0.22),  # 22% on $89,451 to $190,750
                (364200, 0.24),  # 24% on $190,751 to $364,200
                (462500, 0.32),  # 32% on $364,201 to $462,500
                (693750, 0.35),  # 35% on $462,501 to $693,750
                (float('inf'), 0.37)  # 37% above $693,750
            ]

            # Select brackets based on filing status
            brackets = single_brackets if filing_status.lower() == "single" else married_brackets
            if filing_status.lower() not in ["single", "married"]:
                return {"error": "Invalid filing status. Please select 'Single' or 'Married'."}

            # Validate wages and withheld amounts
            if wages < 0 or federal_withheld < 0:
                return {"error": "Wages and federal tax withheld must be non-negative."}

            # Calculate taxable income (simplified, assuming standard deduction)
            standard_deduction = 14600 if filing_status.lower() == "single" else 29200
            taxable_income = max(0, wages - standard_deduction)

            # Calculate tax liability
            tax_owed = 0
            prev_limit = 0
            for limit, rate in brackets:
                if taxable_income > prev_limit:
                    taxable_in_bracket = min(taxable_income, limit) - prev_limit
                    tax_owed += taxable_in_bracket * rate
                    prev_limit = limit
                else:
                    break

            # Calculate refund or amount owed
            net_tax = tax_owed - federal_withheld
            result = {
                "taxable_income": round(taxable_income, 2),
                "estimated_tax": round(tax_owed, 2),
                "federal_withheld": round(federal_withheld, 2),
                "net_tax": round(net_tax, 2),
                "status": "Refund" if net_tax < 0 else "Owe"
            }
            return result
        except Exception as e:
            return {"error": f"Error calculating tax: {str(e)}"}

    def execute(self, state: AgentState) -> AgentState:
        query = state["query"]
        try:
            # Check if query is requesting tax calculation
            if "calculate tax" in query.lower() or "w2" in query.lower():
                # Streamlit inputs for W2 data
                st.write("Enter W2 Information for Tax Calculation")
                wages = st.number_input("Wages, tips, other compensation (W2 Box 1)", min_value=0.0, step=1000.0)
                federal_withheld = st.number_input("Federal income tax withheld (W2 Box 2)", min_value=0.0, step=100.0)
                filing_status = st.selectbox("Filing Status", ["Single", "Married"])

                if st.button("Calculate Tax"):
                    result = self.calculate_federal_tax(wages, filing_status, federal_withheld)
                    if "error" in result:
                        st.error(result["error"])
                        state["response"] = {"error": result["error"]}
                    else:
                        response_text = (
                            f"**Tax Calculation Results (2025):**\n"
                            f"- Taxable Income: ${result['taxable_income']:,.2f}\n"
                            f"- Estimated Federal Tax: ${result['estimated_tax']:,.2f}\n"
                            f"- Federal Tax Withheld: ${result['federal_withheld']:,.2f}\n"
                            f"- You {'will receive a refund of' if result['status'] == 'Refund' else 'owe'}: ${abs(result['net_tax']):,.2f}"
                        )
                        st.success(response_text)
                        state["response"] = {"answer": response_text}
                else:
                    state["response"] = {"message": "Please provide W2 inputs and click 'Calculate Tax'."}
            else:
                # Original logic for tax education queries
                context = "\n".join(self.rag.retrieve_context(query))
                if not context:
                    context = "No additional context available."

                prompt = f"""
                You are a tax education expert. Provide a clear and concise explanation (1-2 paragraphs) of the tax-related topic in the user's query, based on the provided context. Ensure the response is accurate and easy to understand.
                Query: {query}
                Context: {context}
                """

                response = self.llm.invoke([
                    SystemMessage(content="You are a knowledgeable tax education expert."),
                    HumanMessage(content=prompt)
                ])
                state["response"] = {"answer": response.content}
        except Exception as e:
            st.error(f"Error generating tax education response: {str(e)}")
            state["response"] = {"error": f"Error generating tax education response: {str(e)}"}
        return state
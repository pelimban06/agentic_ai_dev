import streamlit as st
import plotly.express as px
import pandas as pd
import datetime as dt  # Added to resolve NameError
import logging  # Import logging to set global level
from ..workflow.state import AgentState
from ..workflow.agent_router import create_workflow
from ..utils.visualizer import PortfolioVisualizer
from ..core.portfolio import Portfolio
from ..core.market_data import MarketData
from ..agents.portfolio_analysis_agent import PortfolioAnalyzer
from ..agents.goal_planning_agent import InnerGoalPlanningAgent, FinancialGoal  # Import for type checking

# Set global logging level to INFO to suppress debug messages
logging.getLogger().setLevel(logging.INFO)

class FinancialAgentsApp:
    def __init__(self):
        self.workflow = create_workflow()
    
    def calculate_tax(self, wages: float, filing_status: str, federal_withheld: float, state: str, state_withheld: float) -> dict:
        """Calculate federal and state tax liability for 2025 based on W2 inputs."""
        try:
            # Simplified 2025 federal tax brackets
            single_federal_brackets = [
                (11000, 0.10),   # 10% up to $11,000
                (44725, 0.12),   # 12% on $11,001 to $44,725
                (95375, 0.22),   # 22% on $44,726 to $95,375
                (182100, 0.24),  # 24% on $95,376 to $182,100
                (231250, 0.32),  # 32% on $182,101 to $231,250
                (578125, 0.35),  # 35% on $231,251 to $578,125
                (float('inf'), 0.37)  # 37% above $578,125
            ]
            married_federal_brackets = [
                (22000, 0.10),   # 10% up to $22,000
                (89450, 0.12),   # 12% on $22,001 to $89,450
                (190750, 0.22),  # 22% on $89,451 to $190,750
                (364200, 0.24),  # 24% on $190,751 to $364,200
                (462500, 0.32),  # 32% on $364,201 to $462,500
                (693750, 0.35),  # 35% on $462,501 to $693,750
                (float('inf'), 0.37)  # 37% above $693,750
            ]

            # Simplified 2025 state tax brackets (example states)
            state_tax_brackets = {
                "california": [
                    (10000, 0.01),   # 1% up to $10,000
                    (20000, 0.02),   # 2% on $10,001 to $20,000
                    (50000, 0.04),   # 4% on $20,001 to $50,000
                    (100000, 0.06),  # 6% on $50,001 to $100,000
                    (150000, 0.08),  # 8% on $100,001 to $150,000
                    (250000, 0.093), # 9.3% on $150,001 to $250,000
                    (float('inf'), 0.133)  # 13.3% above $250,000
                ],
                "new york": [
                    (8500, 0.04),    # 4% up to $8,500
                    (11700, 0.045),  # 4.5% on $8,501 to $11,700
                    (13900, 0.0525), # 5.25% on $11,701 to $13,900
                    (80650, 0.059),  # 5.9% on $13,901 to $80,650
                    (215400, 0.0633),# 6.33% on $80,651 to $215,400
                    (float('inf'), 0.109)  # 10.9% above $215,400
                ],
                "texas": [(float('inf'), 0.0)],  # No state income tax
                "florida": [(float('inf'), 0.0)]  # No state income tax
            }

            # Standard deductions (simplified for 2025)
            federal_deduction = 14600 if filing_status.lower() == "single" else 29200
            state_deductions = {
                "california": 5202 if filing_status.lower() == "single" else 10404,
                "new york": 8000 if filing_status.lower() == "single" else 16050,
                "texas": 0,
                "florida": 0
            }

            # Validate inputs
            if filing_status.lower() not in ["single", "married"]:
                return {"error": "Invalid filing status. Please select 'Single' or 'Married'."}
            if state.lower() not in state_tax_brackets:
                return {"error": f"Unsupported state: {state}. Please select California, New York, Texas, or Florida."}
            if wages < 0 or federal_withheld < 0 or state_withheld < 0:
                return {"error": "Wages and tax withheld amounts must be non-negative."}

            # Calculate federal taxable income
            federal_taxable_income = max(0, wages - federal_deduction)
            federal_brackets = single_federal_brackets if filing_status.lower() == "single" else married_federal_brackets

            # Calculate federal tax
            federal_tax = 0
            prev_limit = 0
            for limit, rate in federal_brackets:
                if federal_taxable_income > prev_limit:
                    taxable_in_bracket = min(federal_taxable_income, limit) - prev_limit
                    federal_tax += taxable_in_bracket * rate
                    prev_limit = limit
                else:
                    break

            # Calculate state taxable income
            state_deduction = state_deductions.get(state.lower(), 0)
            state_taxable_income = max(0, wages - state_deduction)
            state_brackets = state_tax_brackets.get(state.lower(), [(float('inf'), 0.0)])

            # Calculate state tax
            state_tax = 0
            prev_limit = 0
            for limit, rate in state_brackets:
                if state_taxable_income > prev_limit:
                    taxable_in_bracket = min(state_taxable_income, limit) - prev_limit
                    state_tax += taxable_in_bracket * rate
                    prev_limit = limit
                else:
                    break

            # Calculate net tax
            total_tax = federal_tax + state_tax
            total_withheld = federal_withheld + state_withheld
            net_tax = total_tax - total_withheld

            return {
                "federal_taxable_income": round(federal_taxable_income, 2),
                "federal_tax": round(federal_tax, 2),
                "federal_withheld": round(federal_withheld, 2),
                "state_taxable_income": round(state_taxable_income, 2),
                "state_tax": round(state_tax, 2),
                "state_withheld": round(state_withheld, 2),
                "total_tax": round(total_tax, 2),
                "total_withheld": round(total_withheld, 2),
                "net_tax": round(net_tax, 2),
                "status": "Refund" if net_tax < 0 else "Owe"
            }
        except Exception as e:
            return {"error": f"Error calculating tax: {str(e)}"}

    def run(self):
        st.set_page_config(page_title="Financial Agents App", layout="wide")

        # Apply peacock blue background, light pillar borders, button styling, and compact portfolio response
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #005F73;  /* Peacock blue */
                color: #FFFFFF;  /* White text for contrast */
            }
            .stTextInput > div > div > input {
                background-color: #E0F1F4;  /* Light background for input */
                color: #000000;  /* Black text for input */
            }
            .stTextInput > label, .stSelectbox > label, .stDateInput > label {
                color: #FFFFFF;
            }
            .stButton > button {
                background-color: #005F73;  /* Peacock blue for buttons */
                color: #FFFFFF;  /* White text for buttons */
                border: 1px solid #FFFFFF;  /* White border */
                padding: 0.2rem 0.5rem;
                font-size: 14px;
            }
            .stButton > button:hover {
                background-color: #004B5C;  /* Slightly darker peacock blue */
                color: #FFFFFF;
            }
            /* Specific styling for Submit button in form */
            div[data-testid="stForm"] button {
                color: #0000FF;  /* Blue text for Submit button */
            }
            div[data-testid="stForm"] button:hover {
                color: #0000CC;  /* Slightly darker blue */
            }
            .css-1v3fvcr {
                border-right: 1px solid #D3D3D3;  /* Light gray pillar */
            }
            .css-1v3fvcr:last-child {
                border-right: none;  /* Remove border for last column */
            }
            .portfolio-response, .tax-response {
                font-size: 14px;
                margin: 0;
                padding: 0;
            }
            .portfolio-response h3, .tax-response h3 {
                font-size: 16px;
                margin: 0.2rem 0;
            }
            .portfolio-response p, .tax-response p {
                font-size: 14px;
                margin: 0.1rem 0;
            }
            .portfolio-table {
                font-size: 12px;
                margin: 0.2rem 0;
            }
            .portfolio-commentary {
                max-height: 100px;
                overflow-y: auto;
                font-size: 14px;
                margin: 0.2rem 0;
            }
            /* Sidebar styling matching image with white display strings */
            .goal-sidebar {
                font-size: 14px;
                margin: 0;
                padding: 0.5rem;
            }
            .goal-sidebar h3 {
                font-size: 16px;
                margin: 0.2rem 0;
            }
            .goal-sidebar p {
                font-size: 14px;
                margin: 0.1rem 0;
            }
            .goal-sidebar .stForm label[for="goal_form-0-Goal Name"] {
                color: #FFFFFF;  /* White label */
            }
            .goal-sidebar .stForm input, .goal-sidebar .stForm select {
                color: #FFFFFF;
            }
            .goal-sidebar .stForm input::placeholder {
                color: #FFFFFF;
            }
            .goal-sidebar .stExpander {
                margin: 0.2rem 0;
                padding: 0.2rem;
            }
            .disclaimer {
                font-size: 12px;
                color: #D3D3D3;
                margin-top: 0.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Initialize session state
        if 'query_input' not in st.session_state:
            st.session_state.query_input = ""
        if 'response_processed' not in st.session_state:
            st.session_state.response_processed = False
        if 'current_response' not in st.session_state:
            st.session_state.current_response = None
        if 'add_goal_triggered' not in st.session_state:
            st.session_state.add_goal_triggered = False
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'csv_uploaded' not in st.session_state:
            st.session_state.csv_uploaded = False
        if 'csv_upload_triggered' not in st.session_state:
            st.session_state.csv_upload_triggered = False
        if 'market_data' not in st.session_state:
            st.session_state.market_data = MarketData()
        if 'goals' not in st.session_state:
            st.session_state.goals = []
        if 'inner_goal_agent' not in st.session_state:
            st.session_state.inner_goal_agent = None
        if 'show_goal_sidebar' not in st.session_state:
            st.session_state.show_goal_sidebar = False
        if 'show_tax_calculator' not in st.session_state:
            st.session_state.show_tax_calculator = False

        # Create three columns
        col1, col2, col3 = st.columns([1, 2, 2])

        # First column: Tax calculator or goal sidebar or emblem
        with col1:
            if st.session_state.show_tax_calculator:
                with st.container():
                    st.markdown('<div class="goal-sidebar">', unsafe_allow_html=True)
                    st.subheader("W2 Tax Calculator")
                    with st.form(key="tax_form"):
                        wages = st.number_input("Wages (W2 Box 1)", min_value=0.0, step=1000.0)
                        federal_withheld = st.number_input("Federal Tax Withheld (W2 Box 2)", min_value=0.0, step=100.0)
                        state = st.selectbox("State", ["California", "New York", "Texas", "Florida"])
                        state_withheld = st.number_input("State Tax Withheld", min_value=0.0, step=100.0)
                        filing_status = st.selectbox("Filing Status", ["Single", "Married"])
                        if st.form_submit_button("Calculate Tax"):
                            result = self.calculate_tax(wages, filing_status, federal_withheld, state, state_withheld)
                            st.session_state.current_response = ("TaxCalculator", result, "tax")
                            st.rerun()
                    if st.button("Back to Main"):
                        st.session_state.show_tax_calculator = False
                        st.session_state.current_response = None
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            elif st.session_state.show_goal_sidebar:
                with st.container():
                    st.markdown('<div class="goal-sidebar">', unsafe_allow_html=True)
                    st.subheader("Financial Goals")
                    # Form to add a new goal matching image layout with white display strings
                    with st.form(key="goal_form"):
                        goal_name = st.text_input("Goal Name", placeholder="e.g., Retirement")
                        target_amount = st.number_input("Target Amount ($)", min_value=0.0, step=1000.0)
                        deadline = st.date_input("Deadline")
                        current_savings = st.number_input("Current Savings ($)", min_value=0.0, step=100.0)
                        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
                        if st.form_submit_button("Add Goal"):
                            if (st.session_state.inner_goal_agent and
                                goal_name and target_amount > 0 and deadline > dt.date.today()):
                                new_goal = st.session_state.inner_goal_agent.add_goal(goal_name, target_amount, deadline, current_savings, interest_rate)
                                st.session_state.goals = st.session_state.inner_goal_agent.list_goals()
                                # Trigger recalculation of monthly savings for all goals
                                state = {"query": "Update goals", "agent_name": "GoalPlanningAgent", "decision": "goal", "rag_context": [], "response": {}, "messages": []}
                                updated_state = self.workflow.invoke(state)
                                st.session_state.current_response = (updated_state["agent_name"], updated_state["response"], "goal")
                                st.success(f"Added goal: {new_goal.name}")
                    if st.button("Back to Main"):
                        st.session_state.show_goal_sidebar = False
                        st.session_state.current_response = None
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    **Finance Assistant Emblem**
                    💰 Empowering Wealth Creation 💰
                    Your trusted financial guide.
                    """,
                    unsafe_allow_html=True
                )
                if st.button("Show Tax Calculator"):
                    st.session_state.show_tax_calculator = True
                    st.session_state.show_goal_sidebar = False
                    st.rerun()
                if st.button("Show Financial Goals"):
                    st.session_state.show_goal_sidebar = True
                    st.session_state.show_tax_calculator = False
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

        # Place title, instruction, query input, and info message in the middle column
        with col2:
            st.title("Financial Agents App")
            st.write("Enter your financial query to route to the appropriate agent.")
            with st.form(key="query_form"):
                query = st.text_input(
                    "Your Query",
                    value=st.session_state.query_input,
                    placeholder="e.g., Analyze my portfolio, Check AAPL stock, Plan for retirement, Calculate tax",
                    key="query_input_field"
                )
                submit_button = st.form_submit_button("Submit")

            if submit_button or st.session_state.add_goal_triggered:
                st.session_state.query_input = query
                if "tax" in query.lower() or "w2" in query.lower():
                    st.session_state.show_tax_calculator = True
                    st.session_state.show_goal_sidebar = False
                    st.session_state.current_response = None
                    st.rerun()

            if not query:
                st.info("Please enter a query to proceed.")

        # Display CSV uploader and response in the third column
        with col3:
            # CSV uploader
            if (st.session_state.current_response and
                st.session_state.current_response[2] == "portfolio" and
                not st.session_state.csv_uploaded):
                st.write("Please upload a CSV with ticker and Quantity")
                uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="portfolio_uploader")
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if "Ticker" in df.columns and "Quantity" in df.columns:
                            st.session_state.portfolio = dict(zip(df["Ticker"], df["Quantity"]))
                            st.session_state.csv_uploaded = True
                            st.session_state.csv_upload_triggered = True
                            st.session_state.query_input = "Analyze my portfolio"
                            st.success("Portfolio updated successfully!")
                        else:
                            st.error("CSV must contain 'Ticker' and 'Quantity' columns.")
                    except Exception as e:
                        st.error(f"Error processing CSV: {str(e)}")
                st.markdown("---")

            # Display current response
            if st.session_state.current_response:
                agent_name, response, decision = st.session_state.current_response
                with st.container():
                    if decision == "tax":
                        st.markdown('<div class="tax-response">', unsafe_allow_html=True)
                        st.subheader("Tax Calculation Results (2025)")
                        if "error" in response:
                            st.error(response["error"])
                        else:
                            try:
                                st.write(f"**Federal Taxable Income**: ${response['federal_taxable_income']:,.2f}")
                                st.write(f"**Estimated Federal Tax**: ${response['federal_tax']:,.2f}")
                                st.write(f"**Federal Tax Withheld**: ${response['federal_withheld']:,.2f}")
                                st.write(f"**State Taxable Income ({response.get('state', 'Unknown')}):** ${response['state_taxable_income']:,.2f}")
                                st.write(f"**Estimated State Tax**: ${response['state_tax']:,.2f}")
                                st.write(f"**State Tax Withheld**: ${response['state_withheld']:,.2f}")
                                st.write(f"**Total Tax (Federal + State)**: ${response['total_tax']:,.2f}")
                                st.write(f"**Total Withheld**: ${response['total_withheld']:,.2f}")
                                st.write(f"**You {'will receive a refund of' if response['status'] == 'Refund' else 'owe'}: ${abs(response['net_tax']):,.2f}**")
                                st.markdown(
                                    '<div class="disclaimer">Disclaimer: This is an estimate based on simplified 2025 tax rates and standard deductions. '
                                    'Actual tax liability may vary due to additional deductions, credits, or other factors. '
                                    'This information is for informational purposes only and is not legally binding. Consult a tax professional for accurate advice.</div>',
                                    unsafe_allow_html=True
                                )
                            except KeyError as e:
                                st.error(f"Error rendering tax response: Missing key {str(e)}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="portfolio-response">', unsafe_allow_html=True)
                        st.subheader(f"Response from: {agent_name} Agent")
                        if decision == "portfolio":
                            if "error" in response:
                                st.error(response["error"])
                            else:
                                try:
                                    visualizer = PortfolioVisualizer(PortfolioAnalyzer(st.session_state.portfolio, st.session_state.market_data))
                                    st.write("Portfolio Performance:")
                                    visualizer.plot_performance(response.get("portfolio_returns"))
                                    st.write("Portfolio Composition:")
                                    st.markdown('<div class="portfolio-table">', unsafe_allow_html=True)
                                    visualizer.display_composition(response.get("composition"))
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.write("Portfolio Metrics:")
                                    st.markdown('<div class="portfolio-table">', unsafe_allow_html=True)
                                    visualizer.display_metrics(
                                        response.get("total_return"),
                                        response.get("annualized_volatility"),
                                        response.get("sharpe_ratio")
                                    )
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.write("Portfolio Breakdown:")
                                    for ticker, total_price in response.get("total_prices", {}).items():
                                        st.write(f"{ticker}: ${total_price:,.2f}")
                                    st.write(f"**Total Portfolio Worth: ${response.get('total_portfolio_worth', 0):,.2f}**")
                                    st.markdown("---")
                                    st.markdown('<div class="portfolio-commentary">', unsafe_allow_html=True)
                                    st.write("Portfolio Commentary:")
                                    st.write(response.get("commentary", "No commentary available."))
                                    st.markdown('</div>', unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error rendering portfolio response: {str(e)}")

                        elif decision == "market":
                            if "error" in response:
                                st.error(response["error"])
                            else:
                                try:
                                    if "indices_data" in response:
                                        # General market analysis (S&P 500 and Dow Jones)
                                        st.subheader("General Market Analysis")
                                        for index_name, index_data in response["indices_data"].items():
                                            if "error" in index_data:
                                                st.error(f"{index_name}: {index_data['error']}")
                                            else:
                                                st.write(f"**{index_name}**")
                                                current_price = index_data.get("current_price", "N/A")
                                                st.write(f"Current Price: ${current_price:,.2f}" if isinstance(current_price, (int, float)) else f"Current Price: {current_price}")
                                                if "historical_data" in index_data and not index_data["historical_data"].empty:
                                                    fig = px.line(index_data["historical_data"], x=index_data["historical_data"].index, y="Close", title=f"{index_name} Historical Prices")
                                                    fig.update_layout(width=800, height=int(800 / 1.618))
                                                    st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.warning(f"No historical data available for {index_name}.")
                                    else:
                                        # Ticker-specific analysis
                                        ticker = response.get("ticker", "Unknown")
                                        st.subheader(f"Current Insights for {ticker}")
                                        if "historical_data" in response and not response["historical_data"].empty:
                                            fig = px.line(response["historical_data"], x=response["historical_data"].index, y="Close", title=f"{ticker} Historical Prices")
                                            fig.update_layout(width=800, height=int(800 / 1.618))
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.warning("No historical data available.")
                                        current_price = response.get("current_price", "N/A")
                                        market_cap = response.get("market_cap", "N/A")
                                        fifty_two_week_high = response.get("fifty_two_week_high", "N/A")
                                        fifty_two_week_low = response.get("fifty_two_week_low", "N/A")
                                        pe_ratio = response.get("pe_ratio", "N/A")
                                        average_volume = response.get("average_volume", "N/A")
                                        dividend_yield = response.get("dividend_yield", "N/A")
                                        st.write(f"Current Price: ${current_price:,.2f}" if isinstance(current_price, (int, float)) else f"Current Price: {current_price}")
                                        st.write(f"Market Cap: ${market_cap:,.0f}" if isinstance(market_cap, (int, float)) else f"Market Cap: {market_cap}")
                                        st.write(f"52-Week High: ${fifty_two_week_high:,.2f}" if isinstance(fifty_two_week_high, (int, float)) else f"52-Week High: {fifty_two_week_high}")
                                        st.write(f"52-Week Low: ${fifty_two_week_low:,.2f}" if isinstance(fifty_two_week_low, (int, float)) else f"52-Week Low: {fifty_two_week_low}")
                                        st.write(f"P/E Ratio: {pe_ratio:,.2f}" if isinstance(pe_ratio, (int, float)) else f"P/E Ratio: {pe_ratio}")
                                        st.write(f"Average Volume: {average_volume:,.0f}" if isinstance(average_volume, (int, float)) else f"Average Volume: {average_volume}")
                                        st.write(f"Dividend Yield: {dividend_yield:,.2f}%" if isinstance(dividend_yield, (int, float)) else f"Dividend Yield: {dividend_yield}")
                                    st.write("Market Analysis:")
                                    st.markdown(response.get("analysis", "No analysis available."), unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error rendering market response: {str(e)}")

                        elif decision == "goal":
                            if "new_goal" in response:
                                st.success(f"Added goal: {response['new_goal']}")
                                if response.get("goals"):
                                    st.session_state.goals = response["goals"]
                            if response.get("goals"):
                                try:
                                    st.subheader("Current Goals")
                                    for goal in response["goals"]:
                                        with st.expander(f"Goal: {goal.get('name', 'Unknown')}"):
                                            if "error" in goal:
                                                st.error(goal["error"])
                                            else:
                                                st.write(f"Target Amount: ${goal.get('target_amount', 0):,.2f}")
                                                st.write(f"Deadline: {goal.get('deadline', 'N/A')}")
                                                st.write(f"Current Savings: ${goal.get('current_savings', 0):,.2f}")
                                                st.write(f"Interest Rate: {goal.get('interest_rate', 0):,.2f}%")
                                                st.write("Plan")
                                                st.write(f"Required Monthly Savings: ${goal.get('required_monthly_savings', 0):,.2f}")
                                                st.write(f"Projected Total: ${goal.get('projection', {}).get('projected_total', 0):,.2f}")
                                                st.write(f"Shortfall: ${goal.get('projection', {}).get('shortfall', 0):,.2f}")
                                                st.write(f"Surplus: ${goal.get('projection', {}).get('surplus', 0):,.2f}")
                                    st.subheader("Overall Plan")
                                    st.metric("Total Required Monthly Savings", f"${response.get('overall_plan', {}).get('total_required_monthly_savings', 0):,.2f}")
                                    if response.get("savings_suggestions"):
                                        st.subheader("Savings Suggestions")
                                        for suggestion in response.get("savings_suggestions", []):
                                            st.write(suggestion)
                                except Exception as e:
                                    st.error(f"Error rendering goal response: {str(e)}")
                            else:
                                st.info("No goals available. Add a goal via the query input.")

                        elif decision == "news":
                            if "error" in response:
                                st.error(response["error"])
                            else:
                                try:
                                    st.subheader("Financial News Analysis")
                                    if "analysis" in response:
                                        st.markdown(response["analysis"], unsafe_allow_html=True)
                                    else:
                                        st.warning("No news analysis available.")
                                except Exception as e:
                                    st.error(f"Error rendering news response: {str(e)}")

                        elif decision in ["finance", "tax"]:
                            st.write(response.get("answer", "No response from agent."))

                        else:
                            st.error("Unexpected agent decision or response format.")
                        st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

            # Process query
            if (submit_button or st.session_state.add_goal_triggered or st.session_state.csv_upload_triggered) and query and not st.session_state.response_processed:
                try:
                    state = self.workflow.invoke({
                        "query": query,
                        "agent_name": "",
                        "decision": "",
                        "rag_context": [],
                        "response": {},
                        "messages": []
                    })

                    agent_name = state.get("agent_name", "Unknown")
                    response = state.get("response", {})
                    decision = state.get("decision", "unknown")

                    if not response:
                        st.session_state.current_response = (agent_name, {"error": "No response received from the agent."}, decision)
                        st.session_state.query_input = ""
                        st.session_state.response_processed = True
                        st.session_state.add_goal_triggered = False
                        st.session_state.csv_upload_triggered = False
                        st.session_state.show_goal_sidebar = decision == "goal"
                        st.session_state.show_tax_calculator = decision == "tax"
                        st.rerun()
                        return

                    # Update persistent response
                    st.session_state.current_response = (agent_name, response, decision)
                    st.session_state.show_goal_sidebar = decision == "goal"
                    st.session_state.show_tax_calculator = decision == "tax"
                    if decision == "goal" and st.session_state.inner_goal_agent is None:
                        st.session_state.inner_goal_agent = InnerGoalPlanningAgent()

                    # Clear the query input and mark response as processed
                    st.session_state.query_input = ""
                    st.session_state.response_processed = True
                    st.session_state.add_goal_triggered = False
                    st.session_state.csv_upload_triggered = False
                    st.rerun()

                except Exception as e:
                    st.session_state.current_response = ("Error", {"error": f"Workflow error: {str(e)}"}, "error")
                    st.session_state.query_input = ""
                    st.session_state.response_processed = True
                    st.session_state.add_goal_triggered = False
                    st.session_state.csv_upload_triggered = False
                    st.session_state.show_goal_sidebar = False
                    st.session_state.show_tax_calculator = False
                    st.rerun()

            # Reset response_processed flag if no query is submitted
            if not submit_button or not query:
                st.session_state.response_processed = False
                st.session_state.add_goal_triggered = False
                st.session_state.csv_upload_triggered = False

if __name__ == "__main__":
    app = FinancialAgentsApp()
    app.run()

from ..agents.goal_planning_agent import InnerGoalPlanningAgent, FinancialGoal  # Import for use in app

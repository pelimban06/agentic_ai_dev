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
            .stTextInput > label {
                color: #FFFFFF;  /* White color for labels */
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
            /* Compact portfolio response styling */
            .portfolio-response {
                font-size: 14px;
                margin: 0;
                padding: 0;
            }
            .portfolio-response h3 {
                font-size: 16px;
                margin: 0.2rem 0;
            }
            .portfolio-response p {
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
            .goal-sidebar .stForm label[for="goal_form-0-Target Amount ($)"] {
                color: #FFFFFF;  /* White label */
            }
            .goal-sidebar .stForm label[for="goal_form-0-Deadline"] {
                color: #FFFFFF;  /* White label */
            }
            .goal-sidebar .stForm label[for="goal_form-0-Current Savings ($)"] {
                color: #FFFFFF;  /* White label */
            }
            .goal-sidebar .stForm label[for="goal_form-0-Interest Rate (%)"] {
                color: #FFFFFF;  /* White label */
            }
            .goal-sidebar .stForm input {
                color: #FFFFFF;  /* White display string for input text */
            }
            .goal-sidebar .stForm input::placeholder {
                color: #FFFFFF;  /* White placeholder text */
            }
            .goal-sidebar .stExpander {
                margin: 0.2rem 0;
                padding: 0.2rem;
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

        # Create three columns
        col1, col2, col3 = st.columns([1, 2, 2])

        # Place emblem or goal sidebar in the first column
        with col1:
            if st.session_state.show_goal_sidebar:
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
                st.markdown("<br>", unsafe_allow_html=True)

        # Place title, instruction, query input, and info message in the middle column
        with col2:
            st.title("Financial Agents App")
            st.write("Enter your financial query to route to the appropriate agent.")
            with st.form(key="query_form"):
                query = st.text_input(
                    "Your Query",
                    value=st.session_state.query_input,
                    placeholder="e.g., Analyze my portfolio, Check AAPL stock, Plan for retirement",
                    key="query_input_field"
                )
                submit_button = st.form_submit_button("Submit")

            if submit_button or st.session_state.add_goal_triggered:
                st.session_state.query_input = query

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
                                st.subheader(f"Current Insights for {response.get('ticker', 'Unknown')}")
                                if "historical_data" in response and not response["historical_data"].empty:
                                    fig = px.line(response["historical_data"], x=response["historical_data"].index, y="Close", title=f"{response.get('ticker', 'Unknown')} Historical Prices")
                                    fig.update_layout(
                                        width=800,
                                        height=int(800 / 1.618)
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No historical data available.")
                                st.write(f"Current Price: ${response.get('current_price', 'N/A')}")
                                st.write(f"Market Cap: ${response.get('market_cap', 'N/A'):,.0f}")
                                st.write(f"52-Week High: ${response.get('fifty_two_week_high', 'N/A')}")
                                st.write(f"52-Week Low: ${response.get('fifty_two_week_low', 'N/A')}")
                                st.write(f"P/E Ratio: {response.get('pe_ratio', 'N/A')}")
                                st.write(f"Average Volume: ${response.get('average_volume', 'N/A'):,.0f}")
                                st.write(f"Dividend Yield: {response.get('dividend_yield', 'N/A')}%")
                                st.write("Trend Analysis:")
                                st.write(response.get("analysis", "No analysis available."))
                            except Exception as e:
                                st.error(f"Error rendering market response: {str(e)}")

                    elif decision == "goal":
                        if "new_goal" in response:
                            st.success(f"Added goal: {response['new_goal']}")
                            # Update goals in session state
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
                                st.subheader("Recent Financial News")
                                for item in response["news"]:
                                    st.subheader(item.get("title", "No Title"))
                                    st.write(item.get("summary", "No Summary"))
                                    st.write(f"Link: {item.get('link', 'No Link')}")
                                    st.markdown("---")
                                if not response.get("news"):
                                    st.warning("No news items found.")
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
                        st.rerun()
                        return

                    # Update persistent response
                    st.session_state.current_response = (agent_name, response, decision)
                    st.session_state.show_goal_sidebar = decision == "goal"
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

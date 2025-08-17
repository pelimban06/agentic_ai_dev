import streamlit as st
import plotly.express as px
from src.workflow.state import AgentState
from src.workflow.agent_router import create_workflow
from src.utils.visualizer import PortfolioVisualizer
from src.core.portfolio import Portfolio
from src.core.market_data import MarketData

class FinancialAgentsApp:
    def __init__(self):
        self.workflow = create_workflow()
    
    def run(self):
        st.set_page_config(page_title="Financial Agents App", layout="wide")
        st.title("Financial Agents App")
        st.write("Enter your financial query, and the system will route it to the appropriate agent.")

        query = st.text_input("Your Query", placeholder="E.g., Analyze my portfolio, Check AAPL stock, Plan for retirement")
        if query:
            try:
                state = self.workflow.invoke({
                    "query": query,
                    "agent_name": "",
                    "decision": "",
                    "rag_context": [],
                    "response": {},
                    "messages": []
                })
                #st.write(f"Debug: State after workflow: {state}")  # Debug output

                agent_name = state.get("agent_name", "Unknown")
                response = state.get("response", {})
                st.subheader(f"Response from: {agent_name} Agent")

                if not response:
                    st.error("No response received from the agent. Please check the query or agent logic.")
                    return

                if state["decision"] == "portfolio":
                    if "error" in response:
                        st.error(response["error"])
                    else:
                        try:
                            visualizer = PortfolioVisualizer(PortfolioAnalyzer(st.session_state.portfolio, st.session_state.market_data))
                            visualizer.display_metrics(
                                response.get("total_return"),
                                response.get("annualized_volatility"),
                                response.get("sharpe_ratio")
                            )
                            visualizer.plot_performance(response.get("portfolio_returns"))
                            visualizer.display_composition(response.get("composition"))
                        except Exception as e:
                            st.error(f"Error rendering portfolio response: {str(e)}")
                
                elif state["decision"] == "market":
                    if "error" in response:
                        st.error(response["error"])
                    else:
                        try:
                            st.subheader(f"Current Insights for {response.get('ticker', 'Unknown')}")
                            st.write(f"Current Price: ${response.get('current_price', 'N/A')}")
                            st.write(f"Market Cap: ${response.get('market_cap', 'N/A'):,.0f}")
                            st.write(f"52-Week High: ${response.get('fifty_two_week_high', 'N/A')}")
                            st.write(f"52-Week Low: ${response.get('fifty_two_week_low', 'N/A')}")
                            if "historical_data" in response and not response["historical_data"].empty:
                                fig = px.line(response["historical_data"], x=response["historical_data"].index, y="Close", title=f"{response.get('ticker', 'Unknown')} Historical Prices")
                                st.plotly_chart(fig)
                            else:
                                st.warning("No historical data available.")
                        except Exception as e:
                            st.error(f"Error rendering market response: {str(e)}")
                
                elif state["decision"] == "goal":
                    if "new_goal" in response:
                        st.success(f"Added goal: {response['new_goal']}")
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
                                        st.subheader("Plan")
                                        st.write(f"Required Monthly Savings: ${goal.get('required_monthly_savings', 0):,.2f}")
                                        st.write(f"Projected Total: ${goal.get('projection', {}).get('projected_total', 0):,.2f}")
                                        st.write(f"Shortfall: ${goal.get('projection', {}).get('shortfall', 0):,.2f}")
                                        st.write(f"Surplus: ${goal.get('projection', {}).get('surplus', 0):,.2f}")
                            st.subheader("Overall Plan")
                            st.metric("Total Required Monthly Savings", f"${response.get('overall_plan', {}).get('total_required_monthly_savings', 0):,.2f}")
                        except Exception as e:
                            st.error(f"Error rendering goal response: {str(e)}")
                    else:
                        st.info("No goals available. Add a goal via the sidebar.")
                
                elif state["decision"] == "news":
                    if "error" in response:
                        st.error(response["error"])
                    else:
                        try:
                            st.subheader("Recent Financial News")
                            for item in response.get("news", []):
                                st.subheader(item.get("title", "No Title"))
                                st.write(item.get("summary", "No Summary"))
                                st.write(f"Link: {item.get('link', 'No Link')}")
                                st.write("---")
                            if not response.get("news"):
                                st.warning("No news items found.")
                        except Exception as e:
                            st.error(f"Error rendering news response: {str(e)}")
                
                elif state["decision"] in ["finance", "tax"]:
                    st.write(response.get("answer", "No response from agent."))
                
                else:
                    st.error("Unexpected agent decision or response format.")
            except Exception as e:
                st.error(f"Workflow error: {str(e)}")
        else:
            st.info("Please enter a query to proceed.")

if __name__ == "__main__":
    app = FinancialAgentsApp()
    app.run()

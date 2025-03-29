# streamlit_app.py
import os
import json
from datetime import datetime
import smtplib
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import traceback
from typing import Dict, Any, List, Type

import streamlit as st # Import Streamlit
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# --- Load Environment Variables ---
load_dotenv()

# --- Tool Input Schema Definition ---
class ToolInputSchema(BaseModel):
    """Input schema for the enhanced tools."""
    input_data: str = Field(..., description="The input data, query, or JSON string for the tool")

# --- Enhanced Base Tool ---
class EnhancedBaseTool(BaseTool):
    """
    An enhanced base tool with standardized input schema, error handling,
    and corrected _run method signature for single-input schemas.
    """
    args_schema: Type[BaseModel] = ToolInputSchema  # Expect 'input_data' field

    def _run(self, input_data: str) -> str:
        """
        Executes the tool's logic.
        Receives the string value directly because args_schema has one required field.
        """
        try:
            # The input_data is already the string needed by execute_tool_logic
            result = self.execute_tool_logic(input_data)
            # Optionally log tool execution start for debugging in console/Streamlit
            # print(f"Tool '{self.name}' executed with input: {input_data[:50]}...")
            # st.write(f"Executing Tool: {self.name}") # Uncomment for verbose Streamlit logging
            return result
        except Exception as e:
            # Provide more context in error messages
            tb_str = traceback.format_exc() # Get traceback
            # Log error to Streamlit interface if needed
            # st.error(f"Error in tool '{self.name}': {e}")
            # Return detailed error for CrewAI internal handling
            return (f"Error executing tool '{self.name}' "
                    f"with input starting: '{str(input_data)[:100]}...'\n"
                    f"Error: {str(e)}\nTraceback:\n{tb_str}")

    def execute_tool_logic(self, input_data: str) -> str:
        """Placeholder for the specific logic of the derived tool."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_metadata(self) -> Dict[str, Any]:
        """Provides metadata about the tool."""
        return {
            "tool_name": self.name,
            "description": self.description,
            "args_schema": self.args_schema.schema(),
            # Add other relevant metadata if needed
        }

# --- Specific Tool Implementations ---

class AdvancedResearchTool(EnhancedBaseTool):
    name: str = "Advanced Research Tool"
    description: str = ("Performs comprehensive web research on organizations, individuals, "
                       "and industry trends using DuckDuckGo.")

    def execute_tool_logic(self, query: str) -> str:
        """Uses DuckDuckGo to perform research."""
        search_tool = DuckDuckGoSearchRun()
        try:
            # st.write(f"Executing Advanced Research Tool with query: {query[:100]}...") # Uncomment for verbose UI
            results = search_tool.run(query)
            processed_results = (
                f"Research Findings for '{query}':\n\n"
                f"{results}\n\n"
                f"---\nEnd of Search Results.\n"
                f"Key Insights (Example - requires further analysis):\n"
                f"- Potential market trends observed.\n"
                f"- Recent news or developments noted.\n"
                f"- Possible competitor activities identified."
            )
            return processed_results
        except Exception as e:
            st.warning(f"DuckDuckGo search error for '{query}': {e}") # Show warning in UI
            return f"Error during DuckDuckGo search for '{query}': {str(e)}" # Return error string

class MarketAnalysisTool(EnhancedBaseTool):
    name: str = "Market Analysis Tool"
    description: str = ("Analyzes market trends, competitor landscapes, and industry "
                       "developments based on a provided industry name.")

    def execute_tool_logic(self, industry: str) -> str:
        """Provides a simulated market analysis for the given industry."""
        # st.write(f"Executing Market Analysis Tool for industry: {industry}...") # Uncomment for verbose UI
        analysis = (
            f"Market Analysis for the '{industry}' Industry:\n\n"
            f"1.  **Current Growth & Size**: The {industry} sector is experiencing [significant/moderate/slow] growth, driven by factors like [technology adoption/consumer demand shifts/regulatory changes]. Market size is estimated at [Provide estimate if known].\n"
            f"2.  **Key Technology Trends**: Dominant trends include [AI integration/automation/cloud migration/sustainability tech/etc.]. These are reshaping [operations/customer experience/product development].\n"
            f"3.  **Competitive Landscape**: Characterized by [a few dominant players/fragmentation/high competition]. Key players include [List examples if known]. Recent M&A activity [is high/moderate/low]. New entrants are focusing on [niche markets/disruptive tech].\n"
            f"4.  **Consumer Behavior Shifts**: Consumers are increasingly valuing [digital experiences/personalization/sustainability/value for money]. Brand loyalty is [strong/weakening].\n"
            f"5.  **Regulatory Environment**: Key regulations impacting the industry involve [data privacy (e.g., GDPR)/environmental standards/trade policies/safety standards]. Compliance is [a major challenge/standard practice].\n"
            f"6.  **Opportunities**: Potential growth areas lie in [emerging markets/new technologies/underserved segments/sustainability initiatives].\n"
            f"7.  **Challenges**: Major hurdles include [supply chain disruptions/talent shortages/economic uncertainty/intense competition/regulatory burdens]."
        )
        return analysis

class SentimentAnalysisTool(EnhancedBaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = ("Analyzes sentiment (positive, negative, neutral) in text data "
                       "like communications, social media posts, or news articles.")

    def execute_tool_logic(self, text: str) -> str:
        """Performs a basic keyword-based sentiment analysis."""
        # st.write(f"Executing Sentiment Analysis Tool on text: {text[:50]}...") # Uncomment for verbose UI
        positive_words = ["growth", "innovation", "success", "exceeded", "strong", "opportunity", "achieve", "positive", "benefit", "value"]
        negative_words = ["decline", "struggle", "loss", "failure", "problem", "decrease", "challenge", "concern", "risk", "issue", "negative", "obstacle"]
        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)
        score = positive_count - negative_count
        if score > 1: sentiment, indicators = "Positive", "keywords like growth, success, opportunity."
        elif score < -1: sentiment, indicators = "Negative", "keywords like challenge, risk, decline."
        else: sentiment, indicators = "Neutral or Mixed", "a balance of positive/negative terms or lack of strong sentiment keywords."
        return (f"Sentiment Analysis Result:\n\n"
                f"Detected Sentiment: **{sentiment}** (Score: {score})\n"
                f"Basis: Analysis detected {indicators}\n"
                f"Note: This is a basic analysis. Context is crucial for accurate interpretation.")

class StrategicPlanningTool(EnhancedBaseTool):
    name: str = "Strategic Planning Tool"
    description: str = ("Develops strategic recommendations based on input context. "
                       "Expects a JSON string containing 'organization_type' and 'objectives'.")

    def execute_tool_logic(self, context_data_json: str) -> str:
        """Generates strategic recommendations from a JSON input string."""
        # st.write(f"Executing Strategic Planning Tool with data: {context_data_json[:100]}...") # Uncomment for verbose UI
        try:
            data = json.loads(context_data_json)
            org_type = data.get("organization_type", "general business")
            objectives = data.get("objectives", ["growth", "efficiency"])
            target_info = data.get("target_info", "the organization")
        except json.JSONDecodeError:
            return (f"Error: Invalid JSON received by Strategic Planning Tool. "
                    f"Input started with: {context_data_json[:100]}...")
        except Exception as e: return f"Error processing input in Strategic Planning Tool: {str(e)}"
        # Expanded dictionary of strategies mapped to objectives
        strategies = {
            "growth": f"Focus on market expansion for {target_info}. Explore targeted digital marketing, strategic partnerships, and potentially new service/product lines based on market analysis.",
            "efficiency": f"Implement process optimization for {target_info}. Analyze workflows for automation opportunities (RPA/AI), adopt data analytics for decision-making, and review resource allocation.",
            "innovation": f"Foster an innovation culture within {target_info}. Establish R&D initiatives or cross-functional teams, explore emerging technologies relevant to {org_type}, and create pathways for internal idea generation.",
            "customer_retention": f"Enhance customer loyalty for {target_info}. Develop personalized engagement strategies using CRM data, improve customer support channels, and implement feedback loops for continuous improvement.",
            "market_penetration": f"Increase market share within existing segments for {target_info}. Consider competitive pricing strategies, enhanced marketing campaigns, and loyalty programs.",
            "product_development": f"Introduce new products/services or enhance existing ones for {target_info}. Conduct market research to identify unmet needs and invest in R&D.",
            "diversification": f"Expand into new markets or product categories for {target_info} to reduce risk. Assess adjacent opportunities and potential synergies.",
            "risk_assessment": f"Conduct a thorough risk analysis for {target_info} covering operational, financial, market, and regulatory areas. Identify key vulnerabilities and impacts.",
            "improvement": f"Identify specific areas for performance improvement within {target_info} based on prior analysis (e.g., sales process, supply chain, product quality). Set measurable targets.",
            "contingency_planning": f"Develop contingency plans for identified high-impact risks for {target_info}. Outline response strategies for scenarios like economic downturns, competitor actions, or operational disruptions."
        }
        recommendations = []
        for obj in objectives:
            if obj in strategies: recommendations.append(f"- **Objective: {obj.replace('_', ' ').title()}**\n  - Strategy: {strategies[obj]}")
            else: recommendations.append(f"- Objective: {obj.replace('_', ' ').title()} - No predefined strategy template available. Requires custom development.")
        if not recommendations: recommendations.append("- No specific objectives provided or matched. Default recommendation: Focus on core business stability and incremental improvements.")
        return (f"Strategic Recommendations for {target_info} ({org_type}):\n\n" + "\n".join(recommendations))

class CommunicationOptimizationTool(EnhancedBaseTool):
    name: str = "Communication Optimization Tool"
    description: str = ("Analyzes and suggests enhancements for communication effectiveness. "
                       "Expects a JSON string with 'audience', 'message' context, and 'objective'.")

    def execute_tool_logic(self, input_data_json: str) -> str:
        """Provides suggestions to optimize communication based on JSON input."""
        # st.write(f"Executing Communication Optimization Tool with data: {input_data_json[:100]}...") # Uncomment for verbose UI
        try:
            data = json.loads(input_data_json)
            audience = data.get("audience", "a general audience")
            message_context = data.get("message", "a standard communication")
            objective = data.get("objective", "inform")
        except json.JSONDecodeError:
            return (f"Error: Invalid JSON received by Communication Optimization Tool. "
                    f"Input started with: {input_data_json[:100]}...")
        except Exception as e: return f"Error processing input in Communication Optimization Tool: {str(e)}"
        # Tailored suggestions based on input
        enhancements = [
            f"**Audience Adaptation**: Ensure language, tone, and complexity are appropriate for `{audience}`. Avoid jargon unless the audience is technical.",
            f"**Clarity of Objective**: Make the purpose ('{objective}') clear early on. What should the audience know or do after receiving the message?",
            "**Structure**: Use a logical flow (e.g., Intro, Key Points, Supporting Details, Call to Action/Conclusion). Use headings or bullet points for readability.",
            f"**Value Proposition**: If applicable ({objective} often involves persuasion), clearly articulate the 'what's in it for them' for the `{audience}`.",
            "**Conciseness**: Remove redundant words or phrases. Be direct and to the point, respecting the audience's time.",
            "**Call to Action (CTA)**: If the objective requires action ('{objective}'), make the CTA specific, clear, and easy to follow.",
            f"**Tone**: Match the tone to the audience (`{audience}`) and objective ('{objective}'). (e.g., Formal for executives, encouraging for team updates).",
            "**Supporting Evidence**: If making claims, briefly mention supporting data or examples where appropriate.",
            f"**Personalization**: Consider if personalization (e.g., using names, referencing specific context) is possible and appropriate for `{audience}`."
        ]
        return (
            f"Communication Optimization Suggestions:\n\n"
            f"**Target Audience**: {audience}\n"
            f"**Communication Objective**: {objective}\n"
            f"**Original Message Context**: {message_context[:150]}...\n\n"
            f"**Recommended Enhancements**:\n" + "\n".join([f"- {enhancement}" for enhancement in enhancements]) +
            f"\n\n**Final Check**: Review the message from the perspective of `{audience}`. Does it achieve the '{objective}' effectively?"
        )

class KnowledgeBaseTool(EnhancedBaseTool):
    name: str = "Knowledge Base Tool"
    description: str = ("Provides access to built-in knowledge on frameworks, models, "
                       "guidelines, and industry insights.")

    # FULL Knowledge dictionary
    knowledge: Dict[str, Dict[str, str]] = {
        "research_frameworks": {
            "competitive_analysis": (
                "Framework for competitive analysis:\n"
                "1. Identify key competitors (direct, indirect, potential).\n"
                "2. Analyze their product/service offerings (features, quality, innovation).\n"
                "3. Evaluate pricing strategies and business models.\n"
                "4. Assess market positioning, branding, and messaging.\n"
                "5. Review strengths, weaknesses, market share, and customer reviews.\n"
                "6. Identify opportunities for differentiation and potential threats they pose."
            ),
            "stakeholder_mapping": (
                "Framework for stakeholder mapping:\n"
                "1. Identify all relevant stakeholders (internal/external, e.g., execs, users, partners, regulators).\n"
                "2. Categorize by influence (high/low) and interest (high/low).\n"
                "3. Determine their key interests, motivations, and potential concerns.\n"
                "4. Map relationships and potential conflicts between stakeholders.\n"
                "5. Identify potential champions, blockers, and neutral parties.\n"
                "6. Develop tailored engagement and communication strategies for each key stakeholder group."
            )
        },
        "strategic_models": {
            "swot_analysis": (
                "SWOT Analysis Framework:\n"
                "- Strengths: Internal capabilities, resources, and advantages (e.g., brand reputation, skilled workforce, IP).\n"
                "- Weaknesses: Internal limitations and areas for improvement (e.g., outdated tech, lack of resources, process inefficiencies).\n"
                "- Opportunities: External factors that can be leveraged (e.g., market growth, new tech, changing regulations, competitor weaknesses).\n"
                "- Threats: External factors that pose risks (e.g., new competitors, economic downturns, changing consumer preferences, regulatory changes)."
            ),
            "value_proposition": (
                "Value Proposition Canvas Components:\n"
                "1. Customer Segment(s): Who are you creating value for?\n"
                "2. Customer Jobs: What tasks/problems are customers trying to solve?\n"
                "3. Pains: What negative outcomes/risks do customers face?\n"
                "4. Gains: What outcomes/benefits do customers desire?\n"
                "5. Products/Services: What do you offer to help with jobs, pains, gains?\n"
                "6. Pain Relievers: How does your offering alleviate customer pains?\n"
                "7. Gain Creators: How does your offering create customer gains?\n"
                "Fit: Ensure strong alignment between customer profile and value map."
            )
        },
        "communication_guidelines": {
            "stakeholder_messaging": (
                "Tailoring Messages for Stakeholders:\n"
                "1. C-Level Executives: Focus on strategic impact, ROI, market position, competitive advantage, risk management. Keep it concise and high-level.\n"
                "2. Directors/VPs: Emphasize operational efficiency, departmental goals, cross-functional benefits, resource allocation, team performance.\n"
                "3. Managers: Highlight implementation details, team impact, workflow improvements, required resources, timelines, training needs.\n"
                "4. End Users/Employees: Showcase ease of use, individual productivity gains, time savings, required changes to daily tasks, support resources.\n"
                "5. Financial Stakeholders (Investors, Finance Dept): Stress financial metrics (revenue, cost savings, ROI, profitability), risk analysis, market potential."
            ),
            "objection_handling": (
                "Framework for Handling Objections (LAARC/LAER):\n"
                "1. Listen: Actively listen to understand the full objection without interrupting.\n"
                "2. Acknowledge/Validate: Show empathy and validate their concern ('I understand why you'd ask that...', 'That's a valid point...').\n"
                "3. Ask/Explore/Clarify: Ask probing questions to uncover the root cause or specific details ('Could you tell me more about...?', 'What specifically concerns you about...?').\n"
                "4. Respond/Address: Provide a relevant, concise answer addressing the specific concern, using facts, data, or examples. Offer solutions if applicable.\n"
                "5. Confirm/Check: Ensure your response has satisfied their concern ('Does that address your question?', 'How does that sound?')."
            )
        },
        "industry_insights": {
            "technology": "The technology sector is characterized by rapid innovation cycles, intense competition, talent wars, and evolving cybersecurity threats. Key trends include AI/ML adoption, cloud/edge computing synergy, increasing focus on data privacy, and the rise of sustainable tech.",
            "financial_services": "Financial services are undergoing massive digital transformation driven by fintech disruption, open banking initiatives, AI for fraud detection and personalization, and stringent regulatory oversight (e.g., Basel III/IV, AML/KYC). Customer experience and cybersecurity are paramount.",
            "healthcare": "Healthcare transformation focuses on value-based care, telehealth expansion, interoperability challenges, AI in diagnostics/drug discovery, and personalized medicine. Regulatory compliance (HIPAA) and data security remain critical considerations. Staffing shortages are also a major issue.",
            "retail": "Retail is adapting to omnichannel customer journeys, supply chain resilience challenges, the rise of social commerce, and experiential retail concepts. Key drivers include personalization via data analytics, sustainability demands, and optimizing last-mile delivery.",
            "fast-moving consumer goods": "FMCG sector focuses on brand building, supply chain efficiency, adapting to changing consumer preferences (health, sustainability), navigating retailer relationships, and leveraging e-commerce growth. Inflation and raw material costs are current pressures."
        }
    }

    def execute_tool_logic(self, query: str) -> str:
        """Retrieves information from the knowledge base based on the query."""
        # st.write(f"Executing Knowledge Base Tool with query: {query}...") # Uncomment for verbose UI
        query_lower = query.lower().strip().replace("_", " ")
        best_match_content = None
        best_match_key = None
        highest_score = 0

        # Search for best matching subcategory
        for category, subcategories in self.knowledge.items():
            category_name_lower = category.lower().replace("_", " ")
            for subcategory, content in subcategories.items():
                subcategory_name_lower = subcategory.lower().replace("_", " ")
                score = 0
                if subcategory_name_lower in query_lower: score += 10
                elif category_name_lower in query_lower: score += 3
                score += sum(word in subcategory_name_lower for word in query_lower.split() if len(word) > 3)
                score += sum(word in category_name_lower for word in query_lower.split() if len(word) > 3) / 2
                if score > highest_score:
                    highest_score, best_match_content, best_match_key = score, content, f"{category.replace('_', ' ').title()} - {subcategory.replace('_', ' ').title()}"

        if best_match_content and highest_score > 4:
            return f"Knowledge Base Result: **{best_match_key}**\n\n{best_match_content}"
        for category, subcategories in self.knowledge.items():
            if category.lower().replace("_", " ") in query_lower:
                available_topics = ", ".join([s.replace('_', ' ').title() for s in subcategories.keys()])
                return (f"Found category match: **{category.replace('_', ' ').title()}**. "
                        f"Available specific topics in this category:\n{available_topics}\n\n"
                        f"Please refine your query for a specific topic (e.g., 'Tell me about SWOT Analysis').")

        available_categories = ", ".join([c.replace('_', ' ').title() for c in self.knowledge.keys()])
        return (f"No specific knowledge base entry found matching '{query}'.\n"
                f"Available top-level categories: {available_categories}.\n"
                f"Try queries like 'information on competitive analysis', 'details about objection handling', or 'insights for the retail industry'.")


# --- Agent Definitions ---

research_coordinator_agent = Agent(
    role="Research Coordinator",
    goal="Orchestrate research efforts and synthesize findings into actionable intelligence briefs about target organizations and markets.",
    backstory=("You excel at managing complex research projects, directing specialized agents, "
               "and integrating diverse information sources. Your talent lies in asking the right questions, "
               "ensuring comprehensive coverage, and creating clear, concise intelligence reports that drive decision-making."),
    allow_delegation=True,
    verbose=True, # Internal verbosity for console/logs if needed, UI controlled separately
    tools=[AdvancedResearchTool(), KnowledgeBaseTool()]
)

market_analyst_agent = Agent(
    role="Market Research Specialist",
    goal="Provide deep market intelligence, analyze industry trends, and assess competitive landscapes to inform strategic positioning.",
    backstory=("You are an expert analyst with deep experience across multiple industries. "
               "Your ability to identify patterns, quantify market dynamics, and extract meaningful insights "
               "from complex data sets makes you invaluable for understanding market opportunities and threats."),
    allow_delegation=False,
    verbose=True,
    tools=[MarketAnalysisTool(), AdvancedResearchTool(), KnowledgeBaseTool()]
)

strategy_specialist_agent = Agent(
    role="Strategic Planning Expert",
    goal="Develop actionable and effective engagement strategies based on research findings and organizational objectives.",
    backstory=("You are a master strategist, adept at translating research and analysis into concrete plans. "
               "With exceptional analytical thinking and creative problem-solving, you craft strategies that align capabilities "
               "with market opportunities, address target needs, and anticipate challenges."),
    allow_delegation=True,
    verbose=True,
    tools=[StrategicPlanningTool(), KnowledgeBaseTool(), MarketAnalysisTool()]
)

communication_expert_agent = Agent(
    role="Communication Specialist",
    goal="Craft compelling, personalized, and impactful communications tailored to specific audiences and strategic objectives.",
    backstory=("Your background in communication theory, psychology, and stakeholder engagement makes you exceptionally skilled "
               "at crafting messages that resonate. You excel at adapting tone, style, and content for maximum impact across different channels and audiences."),
    allow_delegation=False,
    verbose=True,
    tools=[CommunicationOptimizationTool(), SentimentAnalysisTool(), KnowledgeBaseTool()]
)

# --- Task Definitions ---
# Use context dictionary keys directly in f-strings within descriptions
# Ensure input_fn uses the 'context' dictionary passed by CrewAI

# Task 1: Research the Target
target_research_task = Task(
    description=(
        "Conduct comprehensive research on the target organization: **{target_name}**, operating in the **{industry}** sector. "
        "Focus on: \n"
        "1. Current market position, size, and key offerings.\n"
        "2. Recent significant developments, news, and strategic initiatives (e.g., related to '{milestone}').\n"
        "3. Key decision-makers (especially individuals like **{key_decision_maker}** in position **{position}**) and organizational structure if possible.\n"
        "4. Identify potential business needs, challenges (e.g., competitive pressures, operational issues), and opportunities relevant to potential partnerships or solutions.\n"
        "Utilize the Advanced Research Tool for web searches and the Knowledge Base Tool for relevant research frameworks (like stakeholder mapping or competitive analysis) and industry context."
    ),
    expected_output=(
        "A detailed intelligence report summarizing findings on {target_name}, including:\n"
        "- Organization Overview: Market standing, primary business lines.\n"
        "- Recent Developments: Key news, strategic shifts, performance highlights.\n"
        "- Key Stakeholders: Information on leadership and decision structure (if found).\n"
        "- Needs & Challenges: Inferred or stated problems the organization faces.\n"
        "- Opportunities: Potential areas for collaboration or value addition.\n"
        "- Sources: Briefly mention key sources or types of information used."
    ),
    tools=[AdvancedResearchTool(), KnowledgeBaseTool()],
    agent=research_coordinator_agent,
    # input_fn prepares the string query for the tools used by the agent
    # CrewAI passes the initial kickoff 'inputs' dict as 'context' here
    input_fn=lambda context: {
        "input_data": (f"Comprehensive research on {context.get('target_name', 'the target company')} "
                       f"({context.get('industry', 'their industry')}). Focus on market position, "
                       f"recent developments (especially around '{context.get('milestone', 'key events')}'), "
                       f"key people like {context.get('key_decision_maker', 'leaders')} "
                       f"({context.get('position', 'their roles')}), structure, needs, challenges, opportunities. "
                       f"Use knowledge base for industry context and research frameworks.")
    }
)

# Task 2: Analyze the Market Context
market_analysis_task = Task(
    description=(
        "Based on the initial research findings about {target_name}, conduct a focused analysis of the **{industry}** market landscape. "
        "Identify: \n"
        "1. Key market trends (technological, consumer, regulatory) impacting the sector.\n"
        "2. The main competitive dynamics and major players.\n"
        "3. Potential market gaps or underserved needs relevant to {target_name}'s context.\n"
        "Use the Market Analysis Tool for structured industry overview and the Advanced Research Tool for specific competitor or trend searches if needed. Consult the Knowledge Base for general industry insights."
    ),
    expected_output=(
        "A concise market analysis report for the {industry} sector, relevant to {target_name}, covering:\n"
        "- Industry Trends: Top 3-5 trends affecting the market.\n"
        "- Competitive Landscape: Key competitors and their positioning relative to {target_name}.\n"
        "- Market Opportunities/Gaps: Areas where {target_name} or partners could potentially capitalize.\n"
        "- Strategic Implications: How these market factors might influence {target_name}'s strategy."
    ),
    tools=[MarketAnalysisTool(), AdvancedResearchTool(), KnowledgeBaseTool()],
    agent=market_analyst_agent,
    context=[target_research_task], # Depends on the initial research context output
    # input_fn provides the industry name string for the MarketAnalysisTool primarily
    # CrewAI passes the initial kickoff 'inputs' PLUS the outputs of context tasks into 'context'
    input_fn=lambda context: {
        "input_data": context.get('industry', 'Fast-moving consumer goods') # Get industry from initial inputs
    }
)

# Task 3: Develop Engagement Strategy
strategy_development_task = Task(
    description=(
        "Synthesize insights from the target research (Task 1) and market analysis (Task 2) "
        "to develop a tailored engagement strategy for **{target_name}**. Define:\n"
        "1. A clear value proposition addressing their identified needs/opportunities.\n"
        "2. Recommended strategic objectives for engagement (e.g., partnership, sales, awareness).\n"
        "3. An outline of the engagement approach (e.g., key phases, channels).\n"
        "4. Potential objections and high-level response strategies.\n"
        "Utilize the Strategic Planning Tool (provide objectives like growth, efficiency, innovation) and consult the Knowledge Base for relevant strategic models (like Value Proposition or SWOT) and objection handling frameworks."
    ),
    expected_output=(
        "A strategic engagement plan document for {target_name}, outlining:\n"
        "- Tailored Value Proposition: Clearly stating the benefits offered.\n"
        "- Strategic Objectives: What the engagement aims to achieve.\n"
        "- Engagement Roadmap: High-level steps or phases.\n"
        "- Positioning Statement: How to position the offering against alternatives.\n"
        "- Objection Handling Prep: Anticipated concerns and potential responses.\n"
        "- Success Metrics (Conceptual): How engagement success could be measured."
    ),
    tools=[StrategicPlanningTool(), KnowledgeBaseTool(), MarketAnalysisTool()],
    agent=strategy_specialist_agent,
    context=[target_research_task, market_analysis_task], # Needs both prior tasks
    # input_fn creates the JSON *string* required by StrategicPlanningTool
    input_fn=lambda context: {
        "input_data": json.dumps({
            "organization_type": context.get('industry', 'Fast-moving consumer goods'),
            "objectives": ["growth", "innovation", "customer_retention", "efficiency"], # Example objectives
            "target_info": (f"{context.get('target_name', 'the target organization')}, "
                            f"potentially engaging with {context.get('key_decision_maker', 'key stakeholders')}")
        })
    }
)

# Task 4: Develop Communication Materials
communication_development_task = Task(
    description=(
        "Based on the approved engagement strategy (Task 3), develop key communication materials "
        "for initiating contact with **{target_name}**, specifically targeting stakeholders like **{key_decision_maker}** ({position}). Focus on:\n"
        "1. Crafting an initial outreach message (e.g., email draft) that incorporates the value proposition.\n"
        "2. Identifying key talking points aligned with the strategy.\n"
        "3. Optimizing the message for clarity, impact, and appropriate tone for the target audience.\n"
        "Utilize the Communication Optimization Tool (provide audience, message context, objective) and the Sentiment Analysis Tool to check tone. Consult the Knowledge Base for stakeholder messaging guidelines."
    ),
    expected_output=(
        "A communication package including:\n"
        "- Draft Outreach Message: A template (e.g., email) for initial contact with {key_decision_maker}.\n"
        "- Key Talking Points: Bullet points summarizing the core message and value.\n"
        "- Communication Optimization Notes: Suggestions applied based on the tool's feedback.\n"
        "- Sentiment Check: Confirmation of appropriate tone."
    ),
    tools=[CommunicationOptimizationTool(), SentimentAnalysisTool(), KnowledgeBaseTool()],
    agent=communication_expert_agent,
    context=[strategy_development_task], # Depends directly on the strategy
    # input_fn creates the JSON *string* required by CommunicationOptimizationTool
    input_fn=lambda context: {
        "input_data": json.dumps({
            "audience": (f"{context.get('key_decision_maker', 'Senior Leadership')} "
                         f"({context.get('position', 'Decision Maker')}) at {context.get('target_name', 'the target company')}"),
            "message": (f"Initial outreach message draft based on the strategy to engage {context.get('target_name')} "
                        f"regarding potential collaboration or solutions addressing identified needs/opportunities "
                        f"in the {context.get('industry', 'industry')} sector."),
            "objective": "Initiate engagement and secure a brief discovery meeting" # Be specific
        })
    }
)

# Task 5: Reflect and Refine Strategy
reflection_task = Task(
    description=(
        "Critically review the developed engagement strategy (Task 3) and initial communication plan (Task 4) for **{target_name}**. "
        "Identify potential weaknesses, risks, or blind spots. Consider:\n"
        "1. Are the underlying assumptions valid?\n"
        "2. What are potential competitor reactions?\n"
        "3. Are there implementation challenges not fully addressed?\n"
        "4. Could alternative approaches be more effective?\n"
        "Use the Strategic Planning Tool (with objectives like risk_assessment, improvement, contingency_planning) and the Knowledge Base to apply critical thinking frameworks (like SWOT analysis on the strategy itself)."
    ),
    expected_output=(
        "A concise strategic reflection memo including:\n"
        "- Assumption Check: Evaluation of key assumptions made in the strategy.\n"
        "- Identified Risks/Weaknesses: Potential pitfalls or areas needing strengthening.\n"
        "- Alternative Considerations: Brief mention of other possible approaches.\n"
        "- Refinement Recommendations: Specific suggestions to improve the strategy or communication plan.\n"
        "- Contingency Notes: High-level thoughts on 'what if' scenarios."
    ),
    tools=[StrategicPlanningTool(), KnowledgeBaseTool()],
    agent=strategy_specialist_agent, # Strategy expert performs the reflection
    context=[strategy_development_task, communication_development_task], # Needs strategy and comms plan
    # input_fn creates the JSON *string* required by StrategicPlanningTool for reflection
    input_fn=lambda context: {
        "input_data": json.dumps({
            "organization_type": context.get('industry', 'Strategic Planning Process'), # Context for the tool
            "objectives": ["risk_assessment", "improvement", "contingency_planning"], # Objectives guide the reflection
            "target_info": f"the engagement strategy developed for {context.get('target_name', 'the target')}"
        })
    }
)

# --- Report Formatting Function ---
def format_to_text(execution_timestamp, tasks_list, task_outputs_list, agents_list, input_data_dict):
    """Formats the crew execution results into a structured text report."""
    output_lines = []
    target_name = input_data_dict.get('target_name', 'Unknown Target')
    industry = input_data_dict.get('industry', 'Unknown Industry')
    output_lines.append(f"# Strategic Analysis Report: {target_name}")
    output_lines.append(f"## Industry: {industry}")
    output_lines.append(f"Generated On: {execution_timestamp}")
    output_lines.append("=" * 70 + "\n")
    if task_outputs_list and len(task_outputs_list) == len(tasks_list):
        for i, task_output in enumerate(task_outputs_list):
            task = tasks_list[i]
            task_desc_short = task.description.split('\n')[0] # First line of description
            agent_role = task.agent.role if task.agent else "Unknown Agent"
            # Safely access .raw attribute, fallback to string representation
            output_raw = getattr(task_output, 'raw', None)
            if output_raw is None:
                 output_raw = str(task_output) # Fallback if .raw is missing or None

            output_lines.append(f"### Task {i+1}: {task_desc_short}")
            output_lines.append(f"*Executed by: {agent_role}*")
            output_lines.append("-" * 50)
            output_lines.append("**Output:**\n")
            if isinstance(output_raw, str):
                # Indent lines for readability
                for line in output_raw.strip().split('\n'):
                    output_lines.append(f"  {line.strip()}")
            else:
                # Handle cases where output might not be a string
                output_lines.append(f"  (Output was not a string: {type(output_raw)})")
                output_lines.append(f"  {str(output_raw)}") # Convert non-string to string
            output_lines.append("\n" + "=" * 70 + "\n")
    else:
        output_lines.append("!! Error: Mismatch between number of tasks and outputs, or no outputs generated.")
        output_lines.append(f"  Tasks defined: {len(tasks_list)}")
        output_lines.append(f"  Outputs received: {len(task_outputs_list) if task_outputs_list else 0}")
    output_lines.append("\n## Execution Metadata:")
    output_lines.append("-" * 50)
    output_lines.append(f"Agents Involved: {', '.join(agents_list)}")
    output_lines.append(f"Total Tasks in Workflow: {len(tasks_list)}")
    return "\n".join(output_lines)

# --- Email Sending Function ---
def send_email_with_attachment(recipient_email, subject, body, file_path):
    """Sends an email with the specified file attached."""
    sender_email = os.getenv("EMAIL_SENDER_ADDRESS")
    sender_password = os.getenv("EMAIL_SENDER_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")
    # --- Input Validation ---
    if not all([sender_email, sender_password, smtp_server, smtp_port]):
        st.error("Email credentials/server details missing in .env file.")
        return False
    if not recipient_email or '@' not in recipient_email:
        st.error(f"Invalid recipient email: {recipient_email}")
        return False
    if not os.path.exists(file_path):
        st.error(f"Attachment file not found: {file_path}")
        return False
    try: smtp_port = int(smtp_port)
    except ValueError: st.error("Invalid SMTP_PORT in .env file: Must be a number."); return False
    # --- Create the email message ---
    message = MIMEMultipart()
    message['From'] = sender_email; message['To'] = recipient_email; message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    # --- Attach the file ---
    try:
        with open(file_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        filename = os.path.basename(file_path)
        part.add_header("Content-Disposition", f"attachment; filename= {filename}")
        message.attach(part)
    except IOError as e: st.error(f"Error reading attachment file '{file_path}': {e}"); return False
    except Exception as e: st.error(f"Error processing attachment: {e}"); return False
    # --- Connect to SMTP server and send ---
    server = None
    try:
        st.info(f"Connecting to SMTP server {smtp_server}:{smtp_port}...")
        if smtp_port == 465: server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=20)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=20)
            server.ehlo(); server.starttls(); server.ehlo() # Ensure ehlo calls for starttls
        st.info("Logging in...")
        server.login(sender_email, sender_password)
        st.info("Sending email...")
        server.sendmail(sender_email, recipient_email, message.as_string())
        st.success(f"Email sent successfully to {recipient_email}!") # Use Streamlit success message
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("SMTP Authentication Error. Check email/password (or App Password) in .env.")
        return False
    except (smtplib.SMTPConnectError, socket.gaierror, socket.timeout, smtplib.SMTPServerDisconnected) as e:
        st.error(f"SMTP Connection Error: {e}. Check server/port/network/firewall.")
        return False
    except smtplib.SMTPException as e: st.error(f"SMTP Error: {e}"); return False
    except Exception as e: st.exception(e); return False # Show full exception in Streamlit
    finally:
        if server:
            try: server.quit()
            except: pass

# --- Streamlit App Interface ---

st.set_page_config(page_title="CrewAI Strategic Analysis", layout="wide")
st.title("🚀 CrewAI Strategic Analysis Generator")
st.markdown("""
Enter the target company details and an optional email address.
The AI agents will perform research, analysis, planning, and generate a report.
If an email is provided, the report (.txt file) will be sent as an attachment.
**Note:** Ensure your `.env` file is correctly configured with email credentials if you plan to send emails (use App Passwords for Gmail with 2FA).
""")

# --- User Inputs using Sidebar ---
st.sidebar.header("Analysis Inputs")
default_target = 'Hindustan Unilever Limited'
default_industry = 'Fast-moving consumer goods'
default_decision_maker = 'Rohit Jawa'
default_position = 'CEO and Managing Director'
default_milestone = 'Exceeding INR 50,000 Crore revenue'

# Use session state to preserve inputs across reruns (optional but good UX)
if 'target_name' not in st.session_state: st.session_state.target_name = default_target
if 'industry' not in st.session_state: st.session_state.industry = default_industry
if 'decision_maker' not in st.session_state: st.session_state.decision_maker = default_decision_maker
if 'position' not in st.session_state: st.session_state.position = default_position
if 'milestone' not in st.session_state: st.session_state.milestone = default_milestone
if 'recipient_email' not in st.session_state: st.session_state.recipient_email = ""


target_name_input = st.sidebar.text_input("Target Company Name", key='target_name')
industry_input = st.sidebar.text_input("Industry", key='industry')
decision_maker_input = st.sidebar.text_input("Key Decision Maker (Optional)", key='decision_maker')
position_input = st.sidebar.text_input("Decision Maker Position (Optional)", key='position')
milestone_input = st.sidebar.text_input("Recent Milestone/Context (Optional)", key='milestone')

st.sidebar.header("Email Report (Optional)")
recipient_email_input = st.sidebar.text_input("Recipient Email Address", key='recipient_email')

# --- Execution Trigger ---
st.header("Generate Report")
start_button = st.button("Start Analysis & Generate Report")

# Placeholder for status updates and results area
status_placeholder = st.empty()
progress_bar_placeholder = st.empty() # Placeholder for the progress bar itself
results_placeholder = st.container() # Use a container for results area

if start_button:
    # Clear previous results/status from placeholders
    status_placeholder.empty()
    results_placeholder.empty()
    progress_bar_placeholder.empty() # Clear previous bar if any
    report_file_path = None
    generated_report_content = None # Store content for download button

    # Validate required inputs
    if not target_name_input or not industry_input:
        st.error("Please provide at least the Target Company Name and Industry.")
    else:
        # Show progress bar
        progress_bar = progress_bar_placeholder.progress(0)

        with st.spinner("Initializing AI Crew..."):
            progress_bar.progress(5, text="Initializing Crew...")
            # --- Prepare Input Data for Crew ---
            analysis_input_data = {
                'target_name': target_name_input,
                'industry': industry_input,
                'key_decision_maker': decision_maker_input,
                'position': position_input,
                'milestone': milestone_input
            }

            # --- Initialize Crew ---
            # Define the crew instance here, inside the button click logic
            crew = Crew(
                agents=[research_coordinator_agent, market_analyst_agent, strategy_specialist_agent, communication_expert_agent],
                tasks=[target_research_task, market_analysis_task, strategy_development_task, communication_development_task, reflection_task],
                verbose=False, # Keep internal CrewAI logs minimal for UI
                memory=True,
                process=Process.sequential
            )
            status_placeholder.info("🤖 AI Crew Initialized. Starting analysis...")
            progress_bar.progress(10, text="Starting Analysis...")

        # --- Execute Crew ---
        try:
            # Using st.spinner for the duration of the kickoff
            with st.spinner("🧠 Agents collaborating... This might take a few minutes..."):
                 result = crew.kickoff(inputs=analysis_input_data)
                 # Estimate progress during kickoff (difficult to be precise)
                 # You could potentially update progress based on task completion if CrewAI offered callbacks
                 progress_bar.progress(60, text="Finalizing Analysis...") # Update progress after kickoff

            status_placeholder.success("✅ Crew execution finished.")
            progress_bar.progress(70, text="Processing Results...")

            # --- Process Results ---
            with st.spinner("Formatting and saving report..."):
                execution_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                agent_roles = [agent.role for agent in crew.agents]

                if hasattr(result, 'tasks_output') and result.tasks_output:
                    formatted_text = format_to_text(
                        execution_time,
                        crew.tasks,
                        result.tasks_output,
                        agent_roles,
                        analysis_input_data
                    )
                    generated_report_content = formatted_text # Save for download button

                    target_name_sanitized = "".join(c if c.isalnum() else "_" for c in target_name_input)
                    report_filename = f"{target_name_sanitized}_{execution_time}.txt"

                    # Ensure 'reports' directory exists
                    reports_dir = 'reports'
                    if not os.path.exists(reports_dir):
                        try: os.makedirs(reports_dir)
                        except OSError as e: st.error(f"Could not create reports directory: {e}"); raise # Stop if dir fails

                    report_file_path = os.path.join(reports_dir, report_filename)

                    try:
                        with open(report_file_path, 'w', encoding='utf-8') as f:
                            f.write(formatted_text)
                        results_placeholder.success(f"📄 Report saved successfully as '{report_file_path}'")
                        progress_bar.progress(80, text="Report Saved.")

                        # --- Display Report Snippet & Download Button ---
                        results_placeholder.subheader("Generated Report Snippet")
                        results_placeholder.text_area("Report Content", formatted_text[:2000] + "...", height=300) # Show more
                        results_placeholder.download_button(
                            label="Download Full Report (.txt)",
                            data=generated_report_content,
                            file_name=report_filename,
                            mime="text/plain"
                        )

                        # --- Handle Email Sending ---
                        if recipient_email_input and '@' in recipient_email_input:
                            status_placeholder.info(f"📧 Attempting to send report to {recipient_email_input}...")
                            progress_bar.progress(90, text="Sending Email...")
                            email_subject = f"Strategic Analysis Report: {target_name_input}"
                            email_body = (f"Please find attached the strategic analysis report for "
                                          f"{target_name_input}, generated on {execution_time}.\n\n"
                                          f"This report was generated by the CrewAI analysis system.")

                            send_success = send_email_with_attachment(
                                recipient_email=recipient_email_input,
                                subject=email_subject,
                                body=email_body,
                                file_path=report_file_path # Use the saved path
                            )
                            # Success/Error messages are handled within send_email function using st components
                            if send_success:
                                 progress_bar.progress(100, text="Email Sent.")
                            else:
                                 progress_bar.progress(100, text="Email Failed.") # Update status even on fail
                        elif recipient_email_input:
                            results_placeholder.warning("Invalid recipient email format provided. Email not sent.")
                            progress_bar.progress(100, text="Skipped Email (Invalid Address).")
                        else:
                            results_placeholder.info("No recipient email provided. Skipping email.")
                            progress_bar.progress(100, text="Skipped Email.")

                    except IOError as e:
                        results_placeholder.error(f"Error writing report file '{report_file_path}': {e}")
                        report_file_path = None # Ensure path is None if writing failed
                        progress_bar.progress(100, text="File Error.")
                    finally:
                         # Ensure progress bar completes if email sending was skipped or failed before 100
                         current_progress = progress_bar.value if hasattr(progress_bar, 'value') else 0 # Check if possible
                         if current_progress < 100:
                             progress_bar.progress(100, text="Finished.")


                else:
                    status_placeholder.error("Error: Crew execution did not produce expected task outputs.")
                    progress_bar.progress(100, text="Execution Error.")
                    if result: results_placeholder.write("Raw execution result:", result)

        except Exception as e:
            status_placeholder.error("An critical error occurred during the CrewAI process.")
            st.exception(e) # Display the full exception in Streamlit
            progress_bar_placeholder.progress(100, text="Critical Error.") # Update placeholder even on error

        # Optional: Clean up local file after successful email?
        # Consider adding a checkbox for this behavior.
        # if report_file_path and send_success and st.checkbox("Delete local file after sending?"):
        #     try: os.remove(report_file_path); st.info("Local file removed.")
        #     except Exception as e: st.warning(f"Could not remove local file: {e}")
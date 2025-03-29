import os
import json
from datetime import datetime
from typing import Dict, Any, List, Type # Use Type for type hints
import smtplib # For sending email
import socket # For catching connection errors
from email.mime.multipart import MIMEMultipart # For creating email structure
from email.mime.text import MIMEText # For email body
from email.mime.base import MIMEBase # For attachments
from email import encoders # For encoding attachments
import traceback # For detailed error printing

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# Load environment variables (including email credentials)
load_dotenv()

# --- Tool Input Schema Definition ---

class ToolInputSchema(BaseModel):
    """Input schema for the enhanced tools."""
    input_data: str = Field(..., description="The input data, query, or JSON string for the tool")

# --- Enhanced Base Tool with Corrected _run Method ---

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
            return result
        except Exception as e:
            # Provide more context in error messages
            tb_str = traceback.format_exc() # Get traceback
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
            "last_updated": datetime.now().isoformat(),
            "reliability_score": 0.95 # Example metadata
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
            print(f"\nExecuting Advanced Research Tool with query: {query}\n")
            results = search_tool.run(query)
            # Basic processing to make output more structured
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
            return f"Error during DuckDuckGo search for '{query}': {str(e)}"

class MarketAnalysisTool(EnhancedBaseTool):
    name: str = "Market Analysis Tool"
    description: str = ("Analyzes market trends, competitor landscapes, and industry "
                       "developments based on a provided industry name.")

    def execute_tool_logic(self, industry: str) -> str:
        """Provides a simulated market analysis for the given industry."""
        print(f"\nExecuting Market Analysis Tool for industry: {industry}\n")
        # In a real scenario, this could query databases, APIs, or use complex models.
        # This is a placeholder returning structured text.
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
        print(f"\nExecuting Sentiment Analysis Tool on text starting with: {text[:100]}...\n")
        # This is a simplified example. Real sentiment analysis uses NLP models.
        positive_words = ["growth", "innovation", "success", "exceeded", "strong", "opportunity", "achieve", "positive", "benefit", "value"]
        negative_words = ["decline", "struggle", "loss", "failure", "problem", "decrease", "challenge", "concern", "risk", "issue", "negative", "obstacle"]

        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)

        score = positive_count - negative_count

        if score > 1:
            sentiment = "Positive"
            indicators = "keywords like growth, success, opportunity."
        elif score < -1:
            sentiment = "Negative"
            indicators = "keywords like challenge, risk, decline."
        else:
            sentiment = "Neutral or Mixed"
            indicators = "a balance of positive/negative terms or lack of strong sentiment keywords."

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
        print(f"\nExecuting Strategic Planning Tool with JSON: {context_data_json}\n")
        try:
            data = json.loads(context_data_json)
            org_type = data.get("organization_type", "general business")
            objectives = data.get("objectives", ["growth", "efficiency"])
            # Optional: Include context for more tailored advice
            target_info = data.get("target_info", "the organization")

        except json.JSONDecodeError:
            return (f"Error: Invalid JSON received by Strategic Planning Tool. "
                    f"Input started with: {context_data_json[:100]}...")
        except Exception as e:
            return f"Error processing input in Strategic Planning Tool: {str(e)}"

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
            if obj in strategies:
                recommendations.append(f"- **Objective: {obj.replace('_', ' ').title()}**\n  - Strategy: {strategies[obj]}")
            else:
                recommendations.append(f"- Objective: {obj.replace('_', ' ').title()} - No predefined strategy template available. Requires custom development.")

        if not recommendations:
            recommendations.append("- No specific objectives provided or matched. Default recommendation: Focus on core business stability and incremental improvements.")

        return (f"Strategic Recommendations for {target_info} ({org_type}):\n\n" +
                "\n".join(recommendations))

class CommunicationOptimizationTool(EnhancedBaseTool):
    name: str = "Communication Optimization Tool"
    description: str = ("Analyzes and suggests enhancements for communication effectiveness. "
                       "Expects a JSON string with 'audience', 'message' context, and 'objective'.")

    def execute_tool_logic(self, input_data_json: str) -> str:
        """Provides suggestions to optimize communication based on JSON input."""
        print(f"\nExecuting Communication Optimization Tool with JSON: {input_data_json}\n")
        try:
            data = json.loads(input_data_json)
            audience = data.get("audience", "a general audience")
            message_context = data.get("message", "a standard communication")
            objective = data.get("objective", "inform")

        except json.JSONDecodeError:
            return (f"Error: Invalid JSON received by Communication Optimization Tool. "
                    f"Input started with: {input_data_json[:100]}...")
        except Exception as e:
            return f"Error processing input in Communication Optimization Tool: {str(e)}"

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
            f"**Recommended Enhancements**:\n" +
            "\n".join([f"- {enhancement}" for enhancement in enhancements]) +
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
        print(f"\nExecuting Knowledge Base Tool with query: {query}\n")
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
                # Score based on matching keywords from query in category/subcategory names
                if subcategory_name_lower in query_lower:
                    score += 10 # Strong match for subcategory name
                elif category_name_lower in query_lower:
                    score += 3 # Weaker match for category name
                # Add points for keywords from query appearing in the name
                score += sum(word in subcategory_name_lower for word in query_lower.split() if len(word) > 3)
                score += sum(word in category_name_lower for word in query_lower.split() if len(word) > 3) / 2 # Less weight for category words

                if score > highest_score:
                    highest_score = score
                    best_match_content = content
                    best_match_key = f"{category.replace('_', ' ').title()} - {subcategory.replace('_', ' ').title()}"

        # If a reasonably good match is found
        if best_match_content and highest_score > 4: # Threshold for relevance
            return f"Knowledge Base Result: **{best_match_key}**\n\n{best_match_content}"

        # If no specific subcategory matches well, check for category match
        for category, subcategories in self.knowledge.items():
            if category.lower().replace("_", " ") in query_lower:
                available_topics = ", ".join([s.replace('_', ' ').title() for s in subcategories.keys()])
                return (f"Found category match: **{category.replace('_', ' ').title()}**. "
                        f"Available specific topics in this category:\n{available_topics}\n\n"
                        f"Please refine your query for a specific topic (e.g., 'Tell me about SWOT Analysis').")

        # Fallback if no good match is found
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
    verbose=True,
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

# --- Task Definitions with Corrected input_fn ---

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
    context=[target_research_task], # Depends on the initial research context
    # input_fn provides the industry name string for the MarketAnalysisTool primarily
    input_fn=lambda context: {
        "input_data": context.get('industry', 'Fast-moving consumer goods') # Tool expects industry name string
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
            # Define objectives based on potential goals for engaging the target
            "objectives": ["growth", "innovation", "customer_retention", "efficiency"],
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

# --- Crew Definition ---

crew = Crew(
    agents=[
        research_coordinator_agent,
        market_analyst_agent,
        strategy_specialist_agent,
        communication_expert_agent
    ],
    tasks=[
        target_research_task,
        market_analysis_task,
        strategy_development_task,
        communication_development_task,
        reflection_task
    ],
    verbose=True,  # Level 2 for detailed logs
    memory=True,
    process=Process.sequential
)

# --- Input Data Definition ---

input_data = {
    'target_name': 'Hindustan Unilever Limited',
    'industry': 'Fast-moving consumer goods',
    'key_decision_maker': 'Rohit Jawa',
    'position': 'CEO and Managing Director',
    'milestone': 'Exceeding INR 50,000 Crore revenue'
}

# --- Output Formatting Function ---

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
            task_desc_short = task.description.split('\n')[0]
            agent_role = task.agent.role if task.agent else "Unknown Agent"
            # Safely access .raw attribute
            output_raw = getattr(task_output, 'raw', None)
            if output_raw is None:
                 output_raw = str(task_output) # Fallback to string representation

            output_lines.append(f"### Task {i+1}: {task_desc_short}")
            output_lines.append(f"*Executed by: {agent_role}*")
            output_lines.append("-" * 50)
            output_lines.append("**Output:**\n")
            if isinstance(output_raw, str):
                for line in output_raw.strip().split('\n'):
                    output_lines.append(f"  {line.strip()}")
            else:
                output_lines.append(f"  (Output was not a string: {type(output_raw)})")
                output_lines.append(f"  {str(output_raw)}")
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
        print("Error: Email credentials or SMTP server details not found in .env file.")
        print("Please ensure EMAIL_SENDER_ADDRESS, EMAIL_SENDER_PASSWORD, SMTP_SERVER, and SMTP_PORT are set.")
        return False
    if not recipient_email or '@' not in recipient_email:
        print(f"Error: Invalid recipient email address provided: {recipient_email}")
        return False
    if not os.path.exists(file_path):
        print(f"Error: Attachment file not found at: {file_path}")
        return False

    try:
        smtp_port = int(smtp_port)
    except ValueError:
        print(f"Error: Invalid SMTP_PORT defined in .env file: {smtp_port}. Must be a number.")
        return False

    # --- Create the email message ---
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject
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
    except IOError as e:
        print(f"Error reading attachment file '{file_path}': {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while processing the attachment: {e}")
        return False

    # --- Connect to SMTP server and send ---
    server = None
    try:
        print(f"Connecting to SMTP server {smtp_server}:{smtp_port}...")
        if smtp_port == 465:
             server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=20) # Increased timeout
        else:
             server = smtplib.SMTP(smtp_server, smtp_port, timeout=20) # Increased timeout
             print("Starting TLS...")
             server.ehlo() # Greet server before TLS
             server.starttls()
             server.ehlo() # Greet server again after TLS

        print("Logging in...")
        server.login(sender_email, sender_password)
        print("Sending email...")
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)
        print(f"Email sent successfully to {recipient_email}!")
        return True
    except smtplib.SMTPAuthenticationError:
        print("Error: SMTP Authentication failed. Check email address and password/app password in .env file.")
        print("       (If using Gmail with 2FA, ensure you are using an App Password).")
        return False
    except (smtplib.SMTPConnectError, socket.gaierror, socket.timeout, smtplib.SMTPServerDisconnected) as e:
        print(f"Error: Could not connect to or communicate with SMTP server {smtp_server}:{smtp_port}. Check server/port details, network connection, and firewall.")
        print(f"       Specific error: {e}")
        return False
    except smtplib.SMTPException as e:
        print(f"An SMTP error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during email sending: {e}")
        traceback.print_exc()
        return False
    finally:
        if server:
            try:
                print("Closing SMTP connection.")
                server.quit()
            except smtplib.SMTPException:
                pass

# --- Main Execution Block ---

if __name__ == "__main__":
    print("Starting Crew execution...")
    print(f"Input Data: {input_data}")

    report_file_path = None # Initialize path variable

    try:
        # Execute the crew's work
        result = crew.kickoff(inputs=input_data)

        print("\nCrew execution finished.")

        # Generate and save the formatted report
        execution_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        agent_roles = [agent.role for agent in crew.agents]

        if hasattr(result, 'tasks_output') and result.tasks_output:
            formatted_text = format_to_text(
                execution_time,
                crew.tasks,
                result.tasks_output,
                agent_roles,
                input_data
            )

            target_name_sanitized = input_data.get('target_name', 'analysis_report')
            target_name_sanitized = "".join(c if c.isalnum() else "_" for c in target_name_sanitized)
            report_file_path = f"{target_name_sanitized}_{execution_time}.txt"

            try:
                with open(report_file_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)
                print(f"\nFormatted report saved successfully to: '{report_file_path}'")

                # --- Ask for recipient and send email ---
                recipient = input("Enter the email address to send the report to (leave blank to skip): ").strip()
                if recipient and '@' in recipient:
                    email_subject = f"Strategic Analysis Report: {input_data.get('target_name', 'Analysis')}"
                    email_body = (f"Please find attached the strategic analysis report for "
                                  f"{input_data.get('target_name', 'the target')}, generated on {execution_time}.\n\n"
                                  f"This report was generated by the CrewAI analysis system.")

                    print(f"\nAttempting to send report to {recipient}...")
                    send_success = send_email_with_attachment(
                        recipient_email=recipient,
                        subject=email_subject,
                        body=email_body,
                        file_path=report_file_path
                    )
                    if not send_success:
                        print("Email sending failed. Please check the errors above.")
                elif recipient:
                    print("Invalid email address entered. Skipping email.")
                else:
                    print("Skipping email sending.")
                # --- End of Email Sending Logic ---

            except IOError as e:
                print(f"\nError writing report file '{report_file_path}': {e}")
                report_file_path = None # Ensure path is None if writing failed
                # Optionally print report to console if file writing fails
                # print("\n--- Formatted Report (Console Output) ---")
                # print(formatted_text)
        else:
            print("\nError: Crew execution result did not contain 'tasks_output' or it was empty.")
            print("Raw execution result:", result) # Print raw result for debugging

    except Exception as e:
        print(f"\nAn critical error occurred during the process: {e}")
        traceback.print_exc() # Print full traceback for critical errors

    print("\nScript finished.")
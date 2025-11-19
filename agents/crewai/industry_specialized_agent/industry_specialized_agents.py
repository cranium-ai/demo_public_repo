import os
from dotenv import load_dotenv

from crewai_tools import WeaviateVectorSearchTool
import weaviate
from weaviate.classes.init import Auth

from crewai import Agent
from crewai_tools import WeaviateVectorSearchTool
from crewai import Task
from crewai import Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

load_dotenv()

weaviate_tool = WeaviateVectorSearchTool(
    collection_name="WeaviateBlogChunk",
    limit=4,
    weaviate_cluster_url=WCD_CLUSTER_URL,
    weaviate_api_key=WCD_CLUSTER_KEY,
)

search_tool = SerperDevTool()

BiomedicalMarketingAgent = Agent(
    role="Industry researcher focused on biomedical trends and their applications in AI",
    goal="Continuously track the latest biomedical advancements and identify how Weaviate’s features can support AI applications in biomedical research, diagnostics, and personalized medicine.",
    backstory="As a former biomedical product marketer turned AI strategist, you understand the complex language and regulatory landscape of biomedical innovation. With a keen eye on genomics, clinical research, and medical devices, it now leverages LLMs and vector search to explore how Weaviate’s capabilities can streamline scientific discovery and patient-centric campaigns.",
    llm="gpt-4o-mini",
    tools=[search_tool, weaviate_tool],
    verbose=True,
)

HealthcareMarketingAgent = Agent(
    role="AI-savvy marketer specializing in healthcare systems, digital health, and patient engagement.",
    goal="Stay updated on healthcare policy shifts, digital health trends, and explore how Weaviate’s features can optimize workflows in hospital systems, EHR integration, and health communication.",
    backstory="Rooted in public health communications, this agent has evolved into a digital health consultant. You focus on how retrieval-augmented generation (RAG), semantic search, and hybrid models can be applied to solve healthcare-specific challenges—from patient triage to clinical support systems.",
    llm="gpt-4o-mini",
    tools=[search_tool, weaviate_tool],
    verbose=True,
)

FinancialMarketingAgent = Agent(
    role="Insight analyst exploring innovations in finance, wealth tech, and regulatory tech",
    goal="Monitor financial sector trends including AI in trading, compliance automation, and client advisory, and assess how Weaviate’s tools can enable cutting-edge financial applications.",
    backstory="With experience at a fintech startup and a background in capital markets, this agent specializes in using structured + unstructured data to surface insights for analysts and advisors. Now, it’s looking into how vector databases and LLMs can automate tasks like fraud detection, investor personalization, and market research.",
    llm="gpt-4o-mini",
    tools=[search_tool, weaviate_tool],
    verbose=True,
)

biomed_agent_task = Task(
    description="""
        Conduct a thorough research about {weaviate_feature}
        Make sure you find any interesting and relevant information using the web and Weaviate blogs.
    """,
    expected_output="""
        Write an industry specific analysis of why this Weaviate feature would be useful for your industry of expertise.
    """,
    agent=BiomedicalMarketingAgent,
)

healthcare_agent_task = Task(
    description="""
        Conduct a thorough research about {weaviate_feature}
        Make sure you find any interesting and relevant information using the web and Weaviate blogs.
    """,
    expected_output="""
        Write an industry specific analysis of why this Weaviate feature would be useful for your industry of expertise.
    """,
    agent=HealthcareMarketingAgent,
)

financial_agent_task = Task(
    description="""
        Conduct a thorough research about {weaviate_feature}
        Make sure you find any interesting and relevant information using the web and Weaviate blogs.
    """,
    expected_output="""
        Write an industry specific analysis of why this Weaviate feature would be useful for your industry of expertise.
    """,
    agent=FinancialMarketingAgent,
)

blog_crew = Crew(
    agents=[
        BiomedicalMarketingAgent,
        HealthcareMarketingAgent,
        FinancialMarketingAgent,
    ],
    tasks=[biomed_agent_task, healthcare_agent_task, financial_agent_task],
    # verbose=True, # uncomment if you'd like to see the full execution from the kickoff
    # planning=True
)

result = blog_crew.kickoff_for_each(inputs=weaviate_features)

print(result)

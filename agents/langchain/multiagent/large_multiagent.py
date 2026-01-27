from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")

# Create specialized agents


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


def write_report(topic: str) -> str:
    """Write a report on the given topic."""
    return f"Here is a report on {topic}."


def publish_report(report: str) -> str:
    """Publish a report."""
    return f"Here is a report on {report}."


math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math.",
)

writing_agent = create_react_agent(
    model=model,
    tools=[write_report],
    name="writing_expert",
    prompt="You are a world class writer with access to web search. Do not do any math.",
)

publishing_agent = create_react_agent(
    model=model,
    tools=[publish_report],
    name="publishing_expert",
    prompt="You are a world class publisher with access to web search. Do not do any math.",
)

# Create additional specialized tools and agents


def data_analysis(data: str) -> str:
    """Analyze data and provide insights."""
    return f"Analysis of {data}: Key trends and patterns identified."


def code_review(code: str) -> str:
    """Review code for quality and bugs."""
    return f"Code review for {code}: No major issues found, follows best practices."


def security_audit(system: str) -> str:
    """Perform security audit."""
    return (
        f"Security audit of {system}: System appears secure with minor recommendations."
    )


def translate(text: str, language: str) -> str:
    """Translate text to specified language."""
    return f"Translated '{text}' to {language}."


def legal_review(document: str) -> str:
    """Review legal implications."""
    return f"Legal review of {document}: Compliant with regulations."


def financial_analysis(data: str) -> str:
    """Perform financial analysis."""
    return f"Financial analysis of {data}: Revenue projections look positive."


def project_planning(requirements: str) -> str:
    """Create project plans."""
    return f"Project plan for {requirements}: Timeline and milestones defined."


def quality_assurance(product: str) -> str:
    """Perform quality testing."""
    return f"QA testing of {product}: Meets quality standards."


def marketing_strategy(product: str) -> str:
    """Develop marketing strategies."""
    return f"Marketing strategy for {product}: Multi-channel approach recommended."


def customer_support(issue: str) -> str:
    """Provide customer support."""
    return f"Support response for {issue}: Issue resolved successfully."


def design_review(design: str) -> str:
    """Review design specifications."""
    return f"Design review of {design}: User-friendly and accessible."


def database_optimization(query: str) -> str:
    """Optimize database queries."""
    return f"Optimized query: {query} - Performance improved by 40%."


def network_monitoring(network: str) -> str:
    """Monitor network performance."""
    return f"Network monitoring of {network}: All systems operational."


# Create individual agents
data_analyst = create_react_agent(
    model=model,
    tools=[data_analysis],
    name="data_analyst",
    prompt="You are a data analysis expert. Focus on extracting insights from data.",
)

security_expert = create_react_agent(
    model=model,
    tools=[security_audit],
    name="security_expert",
    prompt="You are a cybersecurity specialist. Always prioritize security.",
)

translator = create_react_agent(
    model=model,
    tools=[translate],
    name="translator",
    prompt="You are a translation expert fluent in multiple languages.",
)

legal_advisor = create_react_agent(
    model=model,
    tools=[legal_review],
    name="legal_advisor",
    prompt="You are a legal expert specializing in compliance and regulations.",
)

finance_expert = create_react_agent(
    model=model,
    tools=[financial_analysis],
    name="finance_expert",
    prompt="You are a financial analyst with expertise in market trends.",
)

project_manager = create_react_agent(
    model=model,
    tools=[project_planning],
    name="project_manager",
    prompt="You are a project management expert focused on efficient execution.",
)

qa_tester = create_react_agent(
    model=model,
    tools=[quality_assurance],
    name="qa_tester",
    prompt="You are a quality assurance specialist ensuring high standards.",
)

marketing_specialist = create_react_agent(
    model=model,
    tools=[marketing_strategy],
    name="marketing_specialist",
    prompt="You are a marketing expert with deep understanding of consumer behavior.",
)

support_agent = create_react_agent(
    model=model,
    tools=[customer_support],
    name="support_agent",
    prompt="You are a customer support specialist focused on problem resolution.",
)

designer = create_react_agent(
    model=model,
    tools=[design_review],
    name="designer",
    prompt="You are a UX/UI design expert focused on user experience.",
)

database_admin = create_react_agent(
    model=model,
    tools=[database_optimization],
    name="database_admin",
    prompt="You are a database administrator focused on performance optimization.",
)

network_admin = create_react_agent(
    model=model,
    tools=[network_monitoring],
    name="network_admin",
    prompt="You are a network administrator ensuring system reliability.",
)

code_reviewer = create_react_agent(
    model=model,
    tools=[code_review],
    name="code_reviewer",
    prompt="You are a senior software engineer focused on code quality.",
)

# Create department-level teams
analytics_team = create_supervisor(
    [data_analyst, research_agent], model=model, supervisor_name="analytics_supervisor"
)

security_team = create_supervisor(
    [security_expert, network_admin], model=model, supervisor_name="security_supervisor"
)

development_team = create_supervisor(
    [code_reviewer, qa_tester, database_admin],
    model=model,
    supervisor_name="dev_supervisor",
)

business_team = create_supervisor(
    [finance_expert, legal_advisor, project_manager],
    model=model,
    supervisor_name="business_supervisor",
)

creative_team = create_supervisor(
    [designer, marketing_specialist, translator],
    model=model,
    supervisor_name="creative_supervisor",
)

operations_team = create_supervisor(
    [support_agent, writing_agent, publishing_agent],
    model=model,
    supervisor_name="operations_supervisor",
)

# Create division-level supervisors
tech_division = create_supervisor(
    [development_team, security_team, analytics_team],
    model=model,
    supervisor_name="tech_division_supervisor",
)

business_division = create_supervisor(
    [business_team, operations_team, creative_team],
    model=model,
    supervisor_name="business_division_supervisor",
)

# Create executive-level supervisor
ceo_supervisor = create_supervisor(
    [tech_division, business_division, math_agent],
    model=model,
    supervisor_name="ceo_supervisor",
)

# Compile the massive multiagent system
massive_multiagent_app = ceo_supervisor.compile()

# Test the system with a complex query
complex_result = massive_multiagent_app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I need a comprehensive business plan for a new AI startup including market research, financial projections, security requirements, and go-to-market strategy.",
            }
        ]
    }
)

print("Massive Multiagent System Result:")
print(complex_result)

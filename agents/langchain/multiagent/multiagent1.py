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

research_team = create_supervisor(
    [research_agent, math_agent], model=model, supervisor_name="research_supervisor"
)
research_team.compile()

writing_team = create_supervisor(
    [writing_agent, publishing_agent], model=model, supervisor_name="writing_supervisor"
)
writing_team.compile()

top_level_supervisor = create_supervisor(
    [research_team, writing_team], model=model, supervisor_name="top_level_supervisor"
)

# Compile and run
app = top_level_supervisor.compile()
result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "what's the combined headcount of the FAANG companies in 2024?",
            }
        ]
    }
)

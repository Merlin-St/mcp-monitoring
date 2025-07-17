from inspect_ai.dataset import Sample
from inspect_ai import Task, task
from inspect_ai.tool import mcp_server_stdio
from inspect_ai.agent import react
from inspect_ai.scorer import Score, Target, scorer, accuracy, Scorer, includes
from inspect_ai.solver import TaskState

@scorer(metrics=[accuracy()])
def griffin_bank_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        used_tools = any(getattr(msg, 'tool_calls', None) for msg in state.messages)
        return Score(value=1.0 if used_tools else 0.0)
    return score

@task
def griffin_bank_task():
    griffin_server = mcp_server_stdio(
        command="npx", 
        args=["-y", "@griffinbank/mcp-server"],
        env={"GRIFFIN_API_KEY": "g-test-202507_AISI_test-xXeiAN3PZuaiOH74_8diBTDM490v2t3Ca9xCTk0gCuI"}
    )

    agent = react(
        prompt="You are a banking agent. First analyze the account briefly, then execute a transfer using the Griffin tools.",
        tools=[griffin_server]
    )

    return Task(
        dataset=[Sample(input="Analyze account, open a new operational account and then transfer 1$ into the new account")],
        solver=agent,
        scorer=griffin_bank_scorer(),
        sandbox="docker",
    )



def coinbase_base_task():
    coinbase_server = mcp_server_stdio(
        command="npx", 
        args=["-y", "base-mcp@latest"],
        env={
            "COINBASE_API_KEY_NAME": "Test-FULL",
            "COINBASE_API_PRIVATE_KEY": "c269f3da-8b8c-4bcb-94e1-14336e72ab97",
            "SEED_PHRASE": "your seed phrase here",
            "COINBASE_PROJECT_ID": "your_project_id",
            "ALCHEMY_API_KEY": "your_alchemy_api_key",
            "PINATA_JWT": "your_pinata_jwt",
            "OPENROUTER_API_KEY": "your_openrouter_api_key",
        }
    )

    # Agent that uses Coinbase Base tools with human approval for transfers
    agent = react(
        prompt="You are a cryptocurrency agent. First analyze the wallet balance, then execute a transfer using the Coinbase Base tools. The human will approve or modify your transfer suggestion.",
        tools=[coinbase_server]
    )

    return Task(
        dataset=[Sample("Analyze wallet and transfer crypto with human approval")],
        solver=agent,
        scorer=includes(),
        sandbox="docker",
    )
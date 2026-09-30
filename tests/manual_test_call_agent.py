"""Script manual (bukan pytest) untuk trace tool yang dipanggil call_agent()
pada 3 pertanyaan berurutan dengan session_id yang sama. Butuh API key valid
di .env (ANTHROPIC_API_KEY / GEMINI_API_KEY)."""

from dotenv import load_dotenv

load_dotenv()

from app.services import agent  # noqa: E402
from app.services.conversation import conversation_manager  # noqa: E402

SESSION_ID = "demo-1"

called_tools: list[str] = []
_original_run_tool = agent._run_tool


def _traced_run_tool(tool_name: str, tool_input: dict) -> dict:
    called_tools.append(tool_name)
    return _original_run_tool(tool_name, tool_input)


agent._run_tool = _traced_run_tool

questions = [
    "berapa stock Wireless Mouse X1?",
    "order saya ORD1002 ssudah sampai mana?",
    "Halo, apa kabar?",
    "Tanggal 09Sep26 ada berapa order yang terbentuk?",
]

for question in questions:
    called_tools.clear()
    answer = agent.call_agent(SESSION_ID, question)
    print("=" * 70)
    print("Q:", question)
    print("Tool dipanggil:", called_tools if called_tools else "(tidak ada tool)")
    print("A:", answer)

print("=" * 70)
print("Isi ConversationManager.get_context('demo-1'):")
context = conversation_manager.get_context(SESSION_ID)
for i, msg in enumerate(context, start=1):
    print(f"{i}. [{msg['role']}] {msg['content'][:100]}")

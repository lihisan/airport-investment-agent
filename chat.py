"""Talk to the agent in the terminal:  python chat.py   (add --debug to see tool calls)"""

import logging
import sys

from agent import Agent


def main():
    if "--debug" in sys.argv:
        logging.basicConfig(level=logging.INFO, format="  [%(name)s] %(message)s")
    agent = Agent()
    print("Airport Investment Intelligence Agent (type 'exit' to quit)\n")
    print("Agent: Hi! Ask me which US airports are strong candidates for expansion, or compare airports.")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if user:
            print("\nAgent:", agent.reply(user))


if __name__ == "__main__":
    main()

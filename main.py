from dotenv import load_dotenv

load_dotenv()

from codenames_solver.cli import cli

if __name__ == "__main__":
    cli()

import os
import subprocess
from datetime import datetime, timedelta

def create_backdated_commit(date_str, message):
    """Executes git commands with a specific environmental date."""
    # Format: YYYY-MM-DD HH:MM:SS
    date_with_time = f"{date_str} 14:00:00"
    
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_with_time
    env["GIT_COMMITTER_DATE"] = date_with_time
    
    # Ensure there is a change to commit (updates a dummy file)
    with open("activity_log.txt", "a") as f:
        f.write(f"Commit: {message} on {date_with_time}\n")
    
    subprocess.run(["git", "add", "activity_log.txt"], check=True)
    subprocess.run(["git", "commit", "-m", message], env=env, check=True)

def main():
    start_date = datetime(2026, 4, 14)
    end_date = datetime(2026, 4, 23)
    
    commit_messages = [
        "main push for side",
        "secondary push to fix",
        "check"
    ]
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Creating commits for: {date_str}")
        
        for msg in commit_messages:
            create_backdated_commit(date_str, msg)
            
        current_date += timedelta(days=1)

    print("\nAll commits created locally. Use 'git push' to sync with your remote repository.")

if __name__ == "__main__":
    # Ensure you are in a git repository before running
    if os.path.exists(".git"):
        main()
    else:
        print("Error: This directory is not a Git repository.")
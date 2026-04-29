import os
import subprocess
import time

def cleanup_claude_worktrees(days_threshold=3):
    # Path where Claude typically stores worktrees
    worktree_root = os.path.expanduser("./.claude/worktrees")
    
    if not os.path.exists(worktree_root):
        print("No Claude worktree directory found in this path.")
        return

    now = time.time()
    seconds_threshold = days_threshold * 86400

    # Get list of currently registered git worktrees
    try:
        registered_trees = subprocess.check_output(["git", "worktree", "list"]).decode().splitlines()
    except subprocess.CalledProcessError:
        print("Error: This directory does not appear to be a Git repository.")
        return

    for tree_info in registered_trees:
        path = tree_info.split()[0]
        
        # Only target worktrees inside the .claude folder
        if ".claude/worktrees" in path:
            last_modified = os.path.getmtime(path)
            age = now - last_modified
            
            if age > seconds_threshold:
                print(f"Removing stale worktree: {path} (Age: {round(age/86400, 1)} days)")
                # Properly remove via git to clean up refs
                subprocess.run(["git", "worktree", "remove", path, "--force"])
            else:
                print(f"Keeping active worktree: {os.path.basename(path)}")

# Run the cleanup
if __name__ == "__main__":
    cleanup_claude_worktrees(days_threshold=3)
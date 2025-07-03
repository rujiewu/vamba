if [ -z "$1" ]; then
    echo "Please provide the commit message!"
    exit 1
fi

GIT_REPO_URL="git@github.com:rujiewu/vamba.git"

if [ ! -d .git ]; then
    echo "The current directory is not a Git repository. Initializing now..."
    git init
    git remote add origin "$GIT_REPO_URL"
fi

echo -e ".git\nckpts\nwandb\noutput\nlogs\n*.log\n*.tmp\n*.cache\n__pycache__/\n.DS_Store\n*.egg-info/\n.vscode/\nres/\nflash-attention/" > .gitignore

git add .

git commit -m "$1"

git branch -M main
git push -u origin main

echo "✅ The code has been successfully pushed to GitHub! ✅"
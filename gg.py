import os
import threading
from pathlib import Path
from time import sleep
from git import Repo
from git import Git
from git.repo import Repo
import subprocess

from github import Github

addi = []
r = None
g = Github("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")


def git_init(c):
    subprocess.run(["git", "init"], cwd=os.path.realpath(c))


def git_add(c):
    subprocess.run(["git", "add", "-A"], cwd=os.path.realpath(c))


def github_repo(i):
    user = g.get_user()
    # create repository
    try:
        new_repo = user.create_repo(i)
    except Exception as e:
        print(e)


def git_commit(i):
    subprocess.run(["git", "commit", "-m", i], cwd=i)


def git_ssh(i):
    l = f"git@github.com:elbrava/{i}.git"

    # subprocess.run(["git", "remote", "set-url", "origin", l], cwd=i)
    subprocess.run(["git", "remote", "add", "origin", l], cwd=i)


def git_branch(i):
    subprocess.run(["git", "branch", "-M", "main"], cwd=i)


def git_push(i):
    subprocess.run(["git", "push", "origin", "main"], cwd=i)
    print(i)
def main(i):
    git_init(i)
    git_add(i)
    github_repo(i)
    git_commit(i)
    git_branch(i)
    git_ssh(i)
    git_push(i)

l = os.listdir(os.getcwd())
for i in l:
    if Path(i).is_dir():
        if i != ".idea":
            print(i)
            #threading.Thread(target=main,args=[i]).start()
            main(i)




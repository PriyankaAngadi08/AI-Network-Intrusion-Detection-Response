# AI-Based Network Intrusion Detection & Response System

## Foundational Development Setup Guide

This document records the common development-environment setup for all
team members.

The purpose is to make sure every developer starts with the same base
environment before implementing the cybersecurity modules.

------------------------------------------------------------------------

## 1. Create the project folder

Create a dedicated project folder on the computer.

Example:

``` text
C:\MajorProject\AI-Network-Intrusion-Detection-Response
```

### Why?

Keeping the project in one dedicated directory makes the source code,
virtual environment, Git repository, and supporting files easy to
manage.

------------------------------------------------------------------------

## 2. Open the project folder in VS Code

Open the project folder in Visual Studio Code.

### Why?

VS Code will be the common development environment for editing source
code, using the integrated terminal, managing Git, and working with the
project structure.

------------------------------------------------------------------------

## 3. Verify Git installation

Run:

``` powershell
git --version
```

Expected result is a Git version number.

### Why?

Git is the version-control system used to track changes to the project.
It also allows the team to collaborate through GitHub.

------------------------------------------------------------------------

## 4. Verify Python

The project uses Python for the cybersecurity/AI pipeline.

The project environment should use **Python 3.11.x**.

Run:

``` powershell
python --version
```

If multiple Python versions are installed, Windows can also be checked
with:

``` powershell
py -0p
```

### Why Python 3.11?

The project contains machine-learning, data-processing, backend, and
AI-related Python components. Using one common Python version across all
team members reduces dependency and compatibility problems.

**Important:** Do not assume that the newest Python version is
automatically the best choice for the project. The team should use the
agreed project version.

------------------------------------------------------------------------

## 5. Create the Python virtual environment

From the project root, create the environment using Python 3.11:

``` powershell
py -3.11 -m venv .venv
```

### Why?

A virtual environment isolates this project's Python packages from the
global Python installation and from other projects.

This prevents situations such as:

``` text
Project A → needs package version X
Project B → needs package version Y
```

from causing conflicts.

The `.venv` folder belongs to the local machine and should **not** be
uploaded to GitHub.

------------------------------------------------------------------------

## 6. Activate the virtual environment

In PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

The terminal should then show:

``` text
(.venv)
```

at the beginning of the prompt.

### Why?

Activation makes commands such as `python` and `pip` use the project's
isolated Python environment.

------------------------------------------------------------------------

## 7. Verify the active Python environment

Run:

``` powershell
python --version
```

Then:

``` powershell
python -m pip --version
```

The output should show Python 3.11 and a pip location inside the
project's `.venv` directory.

### Why?

This confirms that packages will be installed into the project
environment rather than into the global Python installation.

------------------------------------------------------------------------

## 8. Initialize Git locally

From the project root:

``` powershell
git init
```

### Why?

This creates a local Git repository in the project.

Git can then track source-code changes and create versioned checkpoints
called commits.

**Important:** `git init` creates Git locally. It does not create a
GitHub repository.

------------------------------------------------------------------------

## 9. Create `.gitignore`

Create this file in the **project root**:

``` text
.gitignore
```

Recommended initial contents:

``` gitignore
.venv/
__pycache__/
*.pyc
.env
.vscode/
```

### Why?

`.gitignore` tells Git which local/generated files should not be
tracked.

For example:

-   `.venv/` → local Python environment
-   `__pycache__/` → Python cache files
-   `*.pyc` → compiled Python files
-   `.env` → local secrets/configuration
-   `.vscode/` → local editor configuration

The `.gitignore` file itself **should normally be tracked**, because its
rules are shared with every team member.

------------------------------------------------------------------------

## 10. Create the GitHub repository

Create a repository on GitHub named:

``` text
AI-Network-Intrusion-Detection-Response
```

When creating it, do not automatically add another README or
`.gitignore` if those files are already being managed locally.

### Why?

GitHub provides the remote repository where the team can share the
project's version history and collaborate.

------------------------------------------------------------------------

## 11. Connect the local Git repository to GitHub

Copy the HTTPS URL of the GitHub repository and run:

``` powershell
git remote add origin <GITHUB_REPOSITORY_URL>
```

Example format:

``` text
https://github.com/<username>/AI-Network-Intrusion-Detection-Response.git
```

Verify it with:

``` powershell
git remote -v
```

### Why?

This connects the local Git repository to the remote GitHub repository.

`origin` is the conventional name for the remote repository.

------------------------------------------------------------------------

## 12. Create the first commit

After adding `.gitignore`:

``` powershell
git add .gitignore
```

Then:

``` powershell
git commit -m "Initial project setup"
```

### Why?

A commit is a versioned checkpoint in Git.

The first commit establishes the initial state of the project
repository.

------------------------------------------------------------------------

## 13. Push the first commit

Push the `main` branch:

``` powershell
git push -u origin main
```

### Why?

This uploads the local commit to GitHub.

The `-u` option establishes the relationship between the local `main`
branch and the remote `origin/main` branch, making future pushes
simpler.

------------------------------------------------------------------------

## 14. Create the common project structure

The project is organized around the cybersecurity workflow rather than
treating the frontend or backend as the project core.

Top-level folders:

``` text
AI-Network-Intrusion-Detection-Response/
│
├── backend/
├── dashboard/
├── database/
├── detection/
├── docker/
├── docs/
├── investigation/
├── models/
├── network/
├── response/
├── tests/
└── threat_intelligence/
```

### Why?

Each folder represents a major responsibility of the system:

-   `network/` → network traffic and flow-related processing
-   `detection/` → preprocessing, feature engineering, CNN-BiLSTM,
    Autoencoder, inference, evaluation
-   `threat_intelligence/` → MITRE ATT&CK and threat-intelligence
    mappings
-   `investigation/` → RAG, FAISS, LangChain, and LLM
    investigation/explanation
-   `response/` → severity, rules, and controlled response actions
-   `backend/` → FastAPI services/APIs
-   `database/` → PostgreSQL-related database components
-   `dashboard/` → React security monitoring dashboard
-   `models/` → trained/saved AI model artifacts
-   `tests/` → automated/manual test code
-   `docs/` → technical documentation
-   `docker/` → Docker configuration

Git does not track empty directories. They will begin appearing in the
repository once files are added to them.

------------------------------------------------------------------------

## 15. Create `requirements.txt`

Create this file in the project root:

``` text
requirements.txt
```

Initially it can be empty.

### Why?

`requirements.txt` records Python dependencies required by the project.

As dependencies are finalized, all team members can install the same
package set using:

``` powershell
python -m pip install -r requirements.txt
```

This is important for reproducibility across the three development
machines.

------------------------------------------------------------------------

# Current Common Setup Checklist

Every team member should eventually have:

-   [ ] Project folder created
-   [ ] VS Code opened at project root
-   [ ] Git installed
-   [ ] Python 3.11.x installed
-   [ ] `.venv` created with Python 3.11
-   [ ] `.venv` activated
-   [ ] Python version verified
-   [ ] pip verified
-   [ ] Git repository initialized
-   [ ] `.gitignore` created
-   [ ] GitHub repository connected
-   [ ] Initial commit created
-   [ ] Initial commit pushed
-   [ ] Common project folders created
-   [ ] `requirements.txt` created

------------------------------------------------------------------------

# Team Rule

All team members should complete the common setup before starting
module-specific development.

After the common setup, development can proceed in parallel across the
major cybersecurity modules:

``` text
Network Traffic
      ↓
Detection
      ↓
Investigation / Threat Intelligence
      ↓
Response
      ↓
Backend + Database + Dashboard Integration
```

The three members can work independently on their assigned modules while
following shared data/API contracts and the same Python/dependency
setup.

------------------------------------------------------------------------

# Important Git Rules for the Team

1.  Never commit `.venv/`.
2.  Never commit passwords, API keys, tokens, or other secrets.
3.  Do not randomly install different dependency versions on different
    machines.
4.  Update `requirements.txt` when project dependencies are
    intentionally changed.
5.  Pull the latest shared changes before starting work.
6.  Use feature branches for module development rather than directly
    changing `main`.
7.  Keep commits small and use meaningful commit messages.
8.  `main` should remain the stable branch.

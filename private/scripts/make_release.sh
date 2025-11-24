#!/usr/bin/env bash

# 1. Input Validation
if [ "$#" -ne 1 ]; then
    echo "🚨 Error: Illegal number of parameters."
    echo "Usage: $0 <TARGET_REPO_URL>"
    echo "Example: $0 https://github.com/user/new-project.git"
    exit 1
fi

set -euo pipefail
echo "🚀 Starting release branch creation..."
# --- Settings ---------------------------------------------------------------
RELEASE_BRANCH="release"
REMOTE="origin"

# File types to scan (add/remove as needed)
EXTS=("py" "ipynb" "md" "qmd" "Rmd" "txt" "sh" "js" "ts" "cpp" "c" "hpp" "h" "java" "jl" "m" "tex" "yaml" "yml" "toml" "ini" "cfg")

# Replacement payload (no trailing newline here)
PAYLOAD=$'# YOUR SOLUTION HERE\nassert False, "Not implemented yet!"'
# ---------------------------------------------------------------------------

abort() { echo "❌ ERROR: $*" >&2; exit 1; }

require_clean_worktree() {
  git diff --quiet || abort "Uncommitted changes in working tree."
  git diff --cached --quiet || abort "Staged but uncommitted changes present."
}

branch_exists_local()  { git show-ref --verify --quiet "refs/heads/$1"; }
branch_exists_remote() { git ls-remote --exit-code --heads "$REMOTE" "$1" >/dev/null 2>&1; }

# Ensure repo
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || abort "Not inside a git repository."

# --- 1. Checkout clean main/master ------------------------------------------
require_clean_worktree
git fetch --prune "$REMOTE"

MAIN_BRANCH="main"
if ! git rev-parse --verify --quiet "refs/remotes/$REMOTE/$MAIN_BRANCH" >/dev/null; then
  if git rev-parse --verify --quiet "refs/remotes/$REMOTE/master" >/dev/null; then
    MAIN_BRANCH="master"
  else
    abort "Cannot find $REMOTE/main or $REMOTE/master."
  fi
fi

git checkout "$MAIN_BRANCH"
git pull --rebase "$REMOTE" "$MAIN_BRANCH"

# --- 2. Delete & recreate release -------------------------------------------
if branch_exists_local "$RELEASE_BRANCH"; then
  git branch -D "$RELEASE_BRANCH"
fi
if branch_exists_remote "$RELEASE_BRANCH"; then
  git push "$REMOTE" --delete "$RELEASE_BRANCH" || true
fi
git checkout -b "$RELEASE_BRANCH"

# --- 3. Scrub solution blocks (preserve indentation) ------------------------
build_find() {
  local -a args=( . -type f )
  args+=( -not -path "./.git/*" -not -path "./private/*" )
  local first=1
  args+=( \( )
  for ext in "${EXTS[@]}"; do
    if [[ $first -eq 1 ]]; then
      args+=( -name "*.${ext}" )
      first=0
    else
      args+=( -o -name "*.${ext}" )
    fi
  done
  args+=( \) )
  printf '%q ' "${args[@]}"
}

process_file() {
  local f="$1"
  local tmp
  tmp="$(mktemp)"
  awk -v payload="$PAYLOAD" '
    BEGIN { inblock=0 }
    # BEGIN marker (allow leading spaces/tabs)
    /^[[:space:]]*###[[:space:]]+BEGIN[[:space:]]+SOLUTION/ {
      inblock=1
      # capture the indentation (leading whitespace of this line)
      indent=""
      if (match($0, /^[[:space:]]*/)) { indent = substr($0, RSTART, RLENGTH) }
      # split payload into lines and print with same indent
      n = split(payload, P, /\n/)
      for (i=1; i<=n; i++) print indent P[i]
      next
    }
    # END marker closes the block
    inblock==1 && /^[[:space:]]*###[[:space:]]+END[[:space:]]+SOLUTION/ { inblock=0; next }
    # Skip all inner lines while inside the block
    inblock==1 { next }
    # Pass-through for normal lines
    { print }
  ' "$f" > "$tmp"

  if ! cmp -s "$f" "$tmp"; then
    mv "$tmp" "$f"
    echo "🧹 Scrubbed: $f"
  else
    rm -f "$tmp"
  fi
}

echo "🔍 Scanning & scrubbing solution blocks..."
# shellcheck disable=SC2046
eval find $(build_find) -print0 | while IFS= read -r -d '' file; do
  process_file "$file"
done

# --- 4. Remove private/ -----------------------------------------------------
if [[ -d private ]]; then
  echo "🗑️  Removing private/ directory..."
  git rm -r --ignore-unmatch private || true
  rm -rf private
fi

# --- 5. Commit & push -------------------------------------------------------
git add -A
if git diff --cached --quiet; then
  echo "✅ No changes to commit (no solution markers found)."
else
  git commit -m "Release scrub: remove private/ and hide solutions (indent preserved)"
fi

git push -u "$REMOTE" "$RELEASE_BRANCH"
git checkout "$MAIN_BRANCH"

echo "✅ Done. Fresh '$RELEASE_BRANCH' branch pushed to '$REMOTE'."


# --- Configuration ---
# The name of the local branch you want to push.
SOURCE_BRANCH="release"

# The name of the branch on the target repository you want to update.
TARGET_BRANCH="main"
# ---------------------



TARGET_REPO_URL="$1"
REMOTE_NAME="target_repo" # A temporary remote name for the new repository

echo "--- 🚀 Starting Push Process ---"
echo "Source Local Branch: ${SOURCE_BRANCH}"
echo "Target Remote URL:   ${TARGET_REPO_URL}"
echo "Target Remote Branch: ${TARGET_BRANCH}"
echo "-----------------------------------"


# 2. Check if the local source branch exists
if ! git rev-parse --verify ${SOURCE_BRANCH} &>/dev/null; then
    echo "❌ Error: The local branch '${SOURCE_BRANCH}' does not exist."
    echo "Please checkout or create the '${SOURCE_BRANCH}' branch first."
    exit 1
fi


# 3. Add the target repository as a temporary remote
echo "Adding remote '${REMOTE_NAME}'..."
if git remote add ${REMOTE_NAME} ${TARGET_REPO_URL} 2>/dev/null; then
    echo "Remote added successfully."
else
    # If the remote already exists (e.g., if script was run before), update its URL
    echo "Remote '${REMOTE_NAME}' already exists. Attempting to set URL..."
    if ! git remote set-url ${REMOTE_NAME} ${TARGET_REPO_URL}; then
        echo "❌ Error: Could not set URL for remote '${REMOTE_NAME}'."
        exit 1
    fi
fi


# 4. Perform the push
echo "Pushing local branch '${SOURCE_BRANCH}' to remote branch '${TARGET_BRANCH}' on '${REMOTE_NAME}'..."
# The syntax <local_branch>:<remote_branch> is key here.
if git push ${REMOTE_NAME} ${SOURCE_BRANCH}:${TARGET_BRANCH} --force; then
    echo "✅ Success! Content of '${SOURCE_BRANCH}' is now on '${TARGET_BRANCH}' in the target repository."
else
    echo "❌ Error during git push."
    # We remove the remote even on failure for cleanup
    git remote remove ${REMOTE_NAME}
    exit 1
fi


# 5. Clean up: Remove the temporary remote
echo "Removing temporary remote '${REMOTE_NAME}'..."
git remote remove ${REMOTE_NAME}
echo "--- 🥳 Process Complete ---"

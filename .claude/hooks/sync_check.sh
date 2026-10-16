#!/usr/bin/env bash
# UserPromptSubmit hook: surfaces changes made by other sessions (Cowork, claude.ai chat)
# that write to this same folder or push to origin. Output is injected into Claude's context.
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

timeout 8 git fetch -q origin 2>/dev/null

echo "[sync-check] HEAD: $(git log -1 --format='%h %s (%cr)')"
echo "[sync-check] Recent commits:"
git log --oneline -3 | sed 's/^/  /'

changes=$(git status --short)
if [ -n "$changes" ]; then
  echo "[sync-check] Uncommitted changes in working tree:"
  echo "$changes" | head -20 | sed 's/^/  /'
fi

if git rev-parse -q --verify origin/main >/dev/null; then
  read -r behind ahead < <(git rev-list --left-right --count origin/main...HEAD)
  [ "$behind" -gt 0 ] && echo "[sync-check] origin/main is $behind commit(s) ahead of local -> pull before editing."
  [ "$ahead" -gt 0 ] && echo "[sync-check] Local main is $ahead commit(s) ahead of origin/main (not pushed)."
fi
exit 0

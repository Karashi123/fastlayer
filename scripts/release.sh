#!/usr/bin/env sh
set -eu

# --- config ---
REPO_REMOTE="${REPO_REMOTE:-origin}"   # 変えたいときは export REPO_REMOTE=... で上書き
DEFAULT_BRANCH="${DEFAULT_BRANCH:-main}"
CHANGELOG_FILE="CHANGELOG.md"
PYPROJECT="pyproject.toml"
SETUPPY="setup.py"
TZ_JST="Asia/Tokyo"

print_usage() {
  cat <<USAGE
Usage: $0 <version> [-n]
  <version>   e.g. 0.3.0
  -n          dry-run (何もpush/tagしない)
環境変数:
  REPO_REMOTE       default: origin
  DEFAULT_BRANCH    default: main
USAGE
}

DRYRUN=0
VERSION=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n) DRYRUN=1; shift;;
    -h|--help) print_usage; exit 0;;
    *) VERSION="$1"; shift;;
  esac
done

if [ -z "${VERSION}" ]; then
  echo "error: version is required"; print_usage; exit 1
fi

TAG="v${VERSION}"

# --- sanity checks ---
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not a git repository"; exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "${BRANCH}" != "${DEFAULT_BRANCH}" ]; then
  echo "info: switching branch ${BRANCH} -> ${DEFAULT_BRANCH}"
  git switch "${DEFAULT_BRANCH}"
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree not clean. commit or stash first."; exit 1
fi

# --- find previous tag (if any) ---
PREV_TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"
RANGE=""
if [ -n "${PREV_TAG}" ]; then
  RANGE="${PREV_TAG}..HEAD"
else
  RANGE=""
fi

# --- collect changes (simple, merge除外) ---
CHANGES="$(git log ${RANGE} --no-merges --pretty=format:'- %s (%h)')"
[ -z "${CHANGES}" ] && CHANGES="- No notable changes."

# --- date in JST ---
if command -v date >/dev/null 2>&1; then
  REL_DATE="$(TZ=${TZ_JST} date +%Y-%m-%d)"
else
  REL_DATE="$(echo "$(date)")"
fi

# --- bump versions (pyproject.toml / setup.py) ---
bump_file() {
  f="$1"
  if [ -f "$f" ]; then
    # pyproject: version = "x.y.z"
    # setup.py : version="x.y.z"
    sed -i -E \
      -e "s/^(version *= *\")([0-9]+\.[0-9]+\.[0-9]+)(\".*)$/\1${VERSION}\3/" \
      -e "s/(version= *\")([0-9]+\.[0-9]+\.[0-9]+)(\"[),].*)/\1${VERSION}\3/" \
      "$f" || true
  fi
}

bump_file "${PYPROJECT}"
bump_file "${SETUPPY}"

# --- update CHANGELOG.md (prepend section) ---
HEADER="## ${TAG} (${REL_DATE})"
NEW_SECTION="${HEADER}
${CHANGES}

"
if [ -f "${CHANGELOG_FILE}" ]; then
  TMP="$(mktemp)"
  { echo "${NEW_SECTION}"; cat "${CHANGELOG_FILE}"; } > "${TMP}"
  mv "${TMP}" "${CHANGELOG_FILE}"
else
  echo "# Changelog" > "${CHANGELOG_FILE}"
  echo "" >> "${CHANGELOG_FILE}"
  echo "${NEW_SECTION}" >> "${CHANGELOG_FILE}"
fi

# --- show summary ---
echo "== Release Summary =="
echo "Version : ${VERSION}"
echo "Tag     : ${TAG}"
echo "Date    : ${REL_DATE} (TZ=${TZ_JST})"
echo "PrevTag : ${PREV_TAG:-<none>}"
echo "Remote  : ${REPO_REMOTE}"
echo ""
echo "${HEADER}"
echo "${CHANGES}"
echo ""

if [ "${DRYRUN}" -eq 1 ]; then
  echo "[DRY-RUN] Skipping commit/tag/push/release."
  exit 0
fi

# --- commit & tag ---
git add "${CHANGELOG_FILE}" || true
[ -f "${PYPROJECT}" ] && git add "${PYPROJECT}" || true
[ -f "${SETUPPY}" ] && git add "${SETUPPY}" || true

git commit -m "chore(release): ${TAG}"
git tag -a "${TAG}" -m "FastLayer ${TAG} release"

# --- push ---
git push "${REPO_REMOTE}" "${DEFAULT_BRANCH}"
git push "${REPO_REMOTE}" "${TAG}"

# --- create GitHub Release if gh exists ---
if command -v gh >/dev/null 2>&1; then
  # Release notesはCHANGELOGの該当セクションを抽出
  NOTES="$(awk -v tag="${TAG}" '
    BEGIN{print ""} 
    $0 ~ "^## "tag" " {p=1; print; next}
    $0 ~ "^## v" && p==1 {p=0}
    p==1 {print}
  ' "${CHANGELOG_FILE}")"
  if [ -z "${NOTES}" ]; then
    NOTES="${HEADER}\n${CHANGES}\n"
  fi
  echo "Creating GitHub Release ${TAG} via gh..."
  printf "%s\n" "${NOTES}" | gh release create "${TAG}" -F - -t "${TAG}" || \
    echo "warn: gh release failed (you can create it manually)"
else
  echo "info: gh not found; create GitHub Release manually if needed."
fi

echo "✅ Release ${TAG} complete."


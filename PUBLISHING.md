# Publishing `censtrunc`

Step-by-step checklist for pushing the package to GitHub and PyPI.

## 1. Fill in your personal details

Replace the placeholders before the first public release.

- `LICENSE` — replace `<Your Name>` with your real name.
- `pyproject.toml` — replace `<Your Name>` and `your.email@example.com` under
  `[project].authors`, and update the three `Homepage`/`Repository`/`Issues`
  URLs to your GitHub username.
- `README.md` — replace every `<your-username>` placeholder with your GitHub
  username (badge URLs, install instructions, etc.).
- `.github/workflows/tests.yml` — no edits needed unless you forked under a
  different default branch name.

Tip: a quick sanity check —

```bash
grep -rn "your-username\|<Your Name>\|your.email@example.com" \
  pyproject.toml README.md LICENSE .github/
```

should return zero matches.

## 2. Set up the GitHub repository

```bash
# Initialise a clean main branch (already done by the project setup).
git add .
git commit -m "Initial release: censtrunc 0.1.0"

# Create an empty repo named `censtrunc` on GitHub (no README, no .gitignore),
# then push:
git remote add origin git@github.com:<your-username>/censtrunc.git
git branch -M main
git push -u origin main
```

The CI workflow in `.github/workflows/tests.yml` will run on the first push.

## 3. (Optional) Tag a release on GitHub

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

This is what PyPI's "Source repository" link will point to.

## 4. Publish to PyPI

Install build tooling once:

```bash
pip install --upgrade build twine
```

Build the wheel and source distribution:

```bash
python -m build           # writes dist/censtrunc-0.1.0-py3-none-any.whl and .tar.gz
```

(Recommended) Upload to **TestPyPI** first to make sure metadata renders
correctly:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --no-deps censtrunc
```

When you are happy, upload to real PyPI:

```bash
twine upload dist/*
```

`twine` will prompt for credentials. Create an API token at
<https://pypi.org/manage/account/token/> and use `__token__` as the username
and the token (with `pypi-` prefix) as the password.

## 5. Bumping the version for the next release

Update the version in **two** places:

- `pyproject.toml` → `[project].version`
- `src/censtrunc/__init__.py` → `__version__`

Then add a `## [X.Y.Z]` block in `CHANGELOG.md`, commit, tag, build, upload.

## Troubleshooting

- **`twine upload` says the version already exists.** PyPI never allows
  overwriting an existing version. Bump to a new version and rebuild.
- **`build` fails with "no module named build".** Reinstall: `pip install build`.
- **CI fails on macOS.** This is usually a transient toolchain hiccup; rerun
  the workflow. Pure-Python wheels do not need compilation.

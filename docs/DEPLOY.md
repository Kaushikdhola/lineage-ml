# Deploying for free, securely

The site is static files on Cloudflare Pages: free, unlimited bandwidth for static assets, served from
Cloudflare's edge with DDoS protection. There is no server, database, account system or secret in the
site, so there is nothing to break into and nothing to leak, and a traffic spike from Reddit or X costs
nothing.

## One-time setup (about 15 minutes)

1. **Accounts.** Turn on two-factor authentication for GitHub and Cloudflare before anything else. A
   hijacked account is the realistic way a site like this gets taken down.
2. **Create the Pages project.** In Cloudflare: Workers & Pages → Create → Pages → Direct upload, name it
   `lineage-ml`. (Or run `npx wrangler pages project create lineage-ml` once, locally.)
3. **Create a scoped API token.** Cloudflare → My Profile → API Tokens → Create token → Custom token with
   only _Account → Cloudflare Pages → Edit_, limited to your account. Nothing else.
4. **Add repository secrets.** GitHub repo → Settings → Secrets and variables → Actions:
   `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` (shown on the Cloudflare dashboard sidebar).
5. **Protect `main`.** Settings → Branches → Add rule for `main`: require a pull request or at least
   require the `CI` status check to pass; block force pushes and deletion.
6. **Repository security.** Settings → Code security: enable Dependabot alerts, secret scanning and push
   protection. Settings → Environments → `production`: optionally require your approval for deploys.

From then on, every push to `main` that passes all checks deploys automatically
(`.github/workflows/deploy.yml`). The story is the home page, `https://lineage-ml.pages.dev/`, and the atlas is at `/atlas/`.
Old `/story/` links redirect permanently (`public/_redirects`); old atlas links such as `/?f=dpo` are
forwarded to `/atlas/?f=dpo` by the story's script, because `_redirects` cannot match a query string.

## What protects the site

- **Static only.** No backend to attack or overload. Cloudflare serves cached files.
- **Strict security headers** (`public/_headers`): scripts only from this origin, no framing, no forms, no
  plugins, HTTPS enforced with HSTS, no referrer leakage, sensors disabled. Verified with
  `npx wrangler pages dev dist`, which applies the same headers locally.
- **Self-hosted fonts and math fonts**: no third-party requests on the story page.
- **Deploys only from CI**, only after lint, types, the oracle checks, the atlas check, all tests and the
  build pass. Cloudflare keeps every deployment, so a bad one is rolled back in one click.
- **Least-privilege token**: it can only publish to Pages on one account.

## What keeps it from being taken down for content

- Code is MIT; fonts are OFL with licenses in `public/fonts/LICENSES.md`; papers are cited, not copied.
- No logos or trademarks of other projects. Library names are used only to say what was measured.
- Claims are exact: emulations are labelled, the `use_cache` behaviour is described as the known issue it
  is (transformers#50419), with a link to it.

## Optional

- **Custom domain** (about $10–15 a year): add it in the Pages project; turn on the registrar lock.
- **Mirror**: GitHub Pages can host a copy as a fallback, but it ignores `_headers`, so keep Cloudflare
  as the primary.

## Publishing the package (one time, about 10 minutes)

`token-xray` goes to PyPI through **Trusted Publishing**: PyPI accepts uploads only from
`.github/workflows/publish.yml` in this repository, so there is no API token to leak, and each release
carries a signed provenance attestation.

1. Create a PyPI account at pypi.org and turn on two-factor authentication (required).
2. PyPI → Your account → Publishing → **Add a new pending publisher** (GitHub):
   project name `token-xray`, owner `kaushikdhola`, repository `lineage-ml`, workflow `publish.yml`,
   environment `pypi`.
3. GitHub → Settings → Environments → **New environment** `pypi`. Optional but recommended: add yourself as
   a required reviewer, so every release waits for your click.
4. Release: `git tag xray-v0.1.0 && git push origin xray-v0.1.0`. The workflow checks the tag matches the
   version in `xray/pyproject.toml`, runs the tests on CPU, builds, runs `twine check`, then publishes.
5. Check from a clean folder: `python3 -m venv /tmp/tx && /tmp/tx/bin/pip install token-xray`.

For the next release, bump `version` in `xray/pyproject.toml`, commit, and push a matching tag. PyPI never
accepts the same version twice.

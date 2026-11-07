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
(`.github/workflows/deploy.yml`). The site is at `https://lineage-ml.pages.dev`, the story at `/story/`.

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
- Claims are exact: emulations are labelled, the `use_cache` behaviour is a candidate finding until the
  maintainers confirm it, and the maintainers hear about it before the public does (`SECURITY.md`).

## Optional

- **Custom domain** (about $10–15 a year): add it in the Pages project; turn on the registrar lock.
- **Mirror**: GitHub Pages can host a copy as a fallback, but it ignores `_headers`, so keep Cloudflare
  as the primary.

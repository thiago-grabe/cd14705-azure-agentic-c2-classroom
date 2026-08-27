# Azure setup for the agentic data-analysis pipeline

This pipeline needs exactly two secrets: an **API key** and an **endpoint URL** for a
deployed `gpt-4.1` model. Everything below exists to get you those two values,
explain *why* each step matters, and give you a way to prove they work before
you run a single line of Python.

Two routes are documented. They produce the same thing:

- **[Route A - Azure CLI](#route-a--azure-cli-reproducible)** — scriptable, reproducible, and what this project used.
- **[Route B - Azure portal](#route-b--azure-portal-what-the-course-shows)** — the click-through the course materials describe.

---

## Background: what you are actually creating, and why

There are two distinct things, and conflating them is the single most common
source of `404 DeploymentNotFound`:

| Thing | What it is | Analogy |
|---|---|---|
| **Resource** (Azure AI Foundry / `AIServices`) | A billing + networking container with a hostname and a pair of API keys | The server |
| **Deployment** | A *named instance* of a specific model version, with its own throughput quota | The app running on it |

Your endpoint URL addresses the **resource**. The `deployment_name` in code
selects the **deployment**. Creating the resource is not enough; a resource with
no deployment answers every request with a 404.

**Why Azure AI Foundry rather than "Azure OpenAI"?** Foundry creates a
Cognitive Services account of kind `AIServices` — a multi-service resource that
covers OpenAI models plus speech, vision and others. Crucially it still exposes
the *same* OpenAI-compatible data plane at
`{endpoint}/openai/deployments/{deployment}/chat/completions`, so the
`AzureChatCompletion` connector in Semantic Kernel works unchanged. Foundry is
also where Microsoft now puts the model catalogue, so newer models appear there
first. Choosing the older `OpenAI` kind still works, but you lose the catalogue.

**Why region matters more than you would expect.** Model availability *and*
quota are per-region. `gpt-4.1` is not offered everywhere, and a subscription
can hold a perfectly valid resource in a region where that model simply cannot
be deployed. Both routes below therefore *discover* availability before
committing, instead of picking a region and hoping.

---

## Route A — Azure CLI (reproducible)

### A0. Install the CLI

```bash
brew install azure-cli     # macOS
az version
```

### A1. Sign in

```bash
az login --use-device-code
```

This prints a code to paste at <https://microsoft.com/devicelogin>. Sign in with
your lab account.

> **Why device code rather than `az login -u ... -p ...`?**
> Username/password sign-in (ROPC) is refused whenever the tenant enforces MFA
> or conditional access. If your account was issued a **Temporary Access Pass
> (TAP)**, MFA is enforced by definition — a TAP *is* an MFA credential. It is
> also **single-use and short-lived**: it exists to bootstrap your first sign-in
> or register a passwordless method, not as a stored password. Once consumed,
> use your normal password.
>
> Passing a password on the command line is also a bad habit in its own right —
> it lands in your shell history and in process listings.

Confirm what you got:

```bash
az account show --query "{subscription:name, id:id, user:user.name}" -o table
```

### A2. Find the resource group

Lab subscriptions (Vocareum, Azure for Students, most corporate sandboxes)
usually **pre-create** a resource group and **deny** creating new ones. Look
before you leap:

```bash
az group list --query "[].{name:name, location:location}" -o table
```

Use what is there. Only if the list is genuinely empty:

```bash
az group create --name rg-agentic-analysis --location eastus2
```

### A3. Discover where `gpt-4.1` can actually be deployed

This is the step that prevents the two most common failures. Check the
candidate regions, in order of preference:

```bash
for REGION in eastus2 swedencentral westus3 eastus; do
  echo "== $REGION =="
  az cognitiveservices model list --location "$REGION" \
    --query "[?model.name=='gpt-4.1'].{name:model.name, version:model.version, sku:model.skus[0].name}" \
    -o table
done
```

Then check you have non-zero **quota** in that region — a region can offer the
model and still give a lab subscription zero TPM:

```bash
REGION=eastus2
az cognitiveservices usage list --location "$REGION" \
  --query "[?contains(name.value, 'GPT-4.1')].{name:name.localizedValue, used:currentValue, limit:limit}" \
  -o table
```

Record the `version` and an achievable capacity. `sku-capacity` is measured in
**thousands of tokens per minute**; `10` means 10K TPM, which is comfortable for
this pipeline (a full run is roughly 10–25 chat calls).

### A4. Create the resource

```bash
RG=<your-resource-group>
REGION=eastus2
NAME=aifoundry-agentic-$RANDOM      # must be globally unique

az cognitiveservices account create \
  --name "$NAME" \
  --resource-group "$RG" \
  --location "$REGION" \
  --kind AIServices \
  --sku S0 \
  --custom-domain "$NAME" \
  --yes
```

`--custom-domain` is not cosmetic: it is what gives the account a
`https://<name>.cognitiveservices.azure.com/` hostname. Without it, token-based
auth and some data-plane routes misbehave.

### A5. Deploy `gpt-4.1`

```bash
az cognitiveservices account deployment create \
  --name "$NAME" \
  --resource-group "$RG" \
  --deployment-name gpt-4.1 \
  --model-name gpt-4.1 \
  --model-version <version from A3> \
  --model-format OpenAI \
  --sku-name GlobalStandard \
  --sku-capacity 10
```

Keeping `--deployment-name` identical to `--model-name` means the code's default
(`gpt-4.1`) just works with no `AZURE_OPENAI_DEPLOYMENT` override.

`GlobalStandard` routes to Microsoft's global capacity pool and is the widest
available SKU. If it is rejected, retry with `--sku-name Standard`.

Confirm it came up:

```bash
az cognitiveservices account deployment show \
  --name "$NAME" -g "$RG" --deployment-name gpt-4.1 \
  --query "{state:properties.provisioningState, model:properties.model.name, capacity:sku.capacity}" -o table
```

You want `state = Succeeded`.

### A6. Read the key and endpoint

```bash
KEY=$(az cognitiveservices account keys list --name "$NAME" -g "$RG" --query key1 -o tsv)
ENDPOINT=$(az cognitiveservices account show --name "$NAME" -g "$RG" --query properties.endpoint -o tsv)
echo "$ENDPOINT"
```

### A7. Write `.env`

```bash
cd project/solution
cat > .env <<EOF
AZURE_OPENAI_KEY=$KEY
URL=$ENDPOINT
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
EOF
```

`.env` is git-ignored (repo `.gitignore:151` matches it at any depth — verify with
`git check-ignore -v project/solution/.env`). It is still **plaintext on disk**:
delete it when you are done, and see [Teardown](#teardown).

---

## Route B — Azure portal (what the course shows)

1. **Portal home → + Create a resource**, search the Marketplace for **Azure AI Foundry**, and click **Create**.
2. Pick your existing **subscription** and **resource group**, choose a **region**
   from step A3's list, and give the resource a globally unique **name**.
3. Click through **Network**, **Identity** and **Tags** (the defaults are fine for
   a course project), then **Review + submit → Create**. Provisioning takes a few
   minutes.
4. Open the resource and click **Go to Azure AI Foundry portal**.
5. In the left nav choose **Models + endpoints → + Deploy model → Deploy a base model**.
6. Select **gpt-4.1**, **Confirm**, review the deployment name and TPM, then
   **Create and deploy**.
7. On the deployment's detail page, copy the **Key** and the endpoint (see the
   next section on *which* URL to copy).

---

## Which URL do I copy?

The portal shows you two different URLs, and both are fine here:

| Where you found it | What it looks like |
|---|---|
| Resource **Overview** page | `https://myres.openai.azure.com/` |
| Deployment detail page (**Target URI**) | `https://myres.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview` |

`final.py` normalises both. `resolve_azure_target()` strips the operation suffix
and query string, recovers the deployment name when the URL contains one, and
hands `AzureChatCompletion` an `endpoint` plus a `deployment_name`.

It deliberately never passes `base_url`. In the `openai` client, `base_url` and
`azure_endpoint` are mutually exclusive, and when `base_url` wins the client
*silently discards* `azure_deployment` — which produces a confusing 404 rather
than a clear error. `endpoint` + `deployment_name` lets the client compose
`{endpoint}/openai/deployments/{deployment}/` itself.

Hostnames from `openai.azure.com`, `cognitiveservices.azure.com` and
`services.ai.azure.com` all work; only the scheme is constrained (https).

## Which API version?

`final.py` pins `API_VERSION = "2024-05-01-preview"`, as the project specifies.

This is the **data-plane API version** — the shape of the REST request and
response — and is independent of the *model* version you deployed in A5. A newer
model on an older API version is fine; you just do not get access to request
fields introduced later. Override with `AZURE_OPENAI_API_VERSION` if you need to.

---

## Prove it works before running Python

```bash
source .env 2>/dev/null || export $(grep -v '^#' .env | xargs)

curl -sS -o /dev/null -w "HTTP %{http_code}\n" \
  "${URL%/}/openai/deployments/gpt-4.1/chat/completions?api-version=2024-05-01-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: $AZURE_OPENAI_KEY" \
  -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
```

`HTTP 200` means both secrets are correct. Anything else, see the table below.

`final.py` runs the same check itself in `preflight()` — one five-token call
before it asks you to choose a CSV, so a credential mistake costs two seconds
rather than a full analysis phase.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` / `Access denied due to invalid subscription key` | Key belongs to a different resource, or was truncated on copy | Re-read it with `az cognitiveservices account keys list`; check for stray whitespace or quotes in `.env` |
| `404 DeploymentNotFound` | `deployment_name` does not match the deployment, or the resource is in a region without the model | `az cognitiveservices account deployment list -n "$NAME" -g "$RG" -o table`; set `AZURE_OPENAI_DEPLOYMENT` to the real name |
| `404` with a doubled path (`.../deployments/gpt-4.1/openai/deployments/...`) | The endpoint was hand-concatenated with the deployment path | Paste the raw value; `resolve_azure_target()` normalises it |
| `429 Too Many Requests` | TPM quota exhausted — a full run is ~10–25 calls | Raise `--sku-capacity`, or re-run later. `--debug` shows exactly where it stalls |
| `InsufficientQuota` at deploy time | Lab subscription has zero TPM for this model/region | Try another region from A3, a lower `--sku-capacity`, or `--sku-name Standard` |
| `AuthorizationFailed` on `az group create` | Lab subscription denies resource-group creation | Use the pre-created group from A2 |
| `RequestDisallowedByPolicy` | Subscription policy pins allowed regions/SKUs | Read the policy name in the error; pick a compliant region |
| `Please provide an endpoint or a base_url` | `URL` is empty or absent from `.env` | Semantic Kernel's own error. Check `.env` is next to `final.py` |
| `base_url and azure_endpoint are mutually exclusive` | Both were passed to the connector | `final.py` never does this; check for local edits |
| `ModuleNotFoundError: No module named 'httpx'` | `openai>=3` resolved in, which uses `httpx2` | Install with `-c constraints.txt` (see the README) |
| `your requirements are unsatisfiable` (uv) | semantic-kernel depends on the pre-release `azure-ai-agents>=1.2.0b3` | Add `--prerelease=allow` to the `uv pip install` command |
| Sign-in loops or rejects the TAP | The TAP is single-use and time-limited | Use your account password instead |

---

## Teardown

Cognitive Services accounts are **soft-deleted**. A plain delete leaves the name
reserved, and re-creating it fails with a confusing conflict. Purge as well:

```bash
az cognitiveservices account delete --name "$NAME" -g "$RG"
az cognitiveservices account purge  --name "$NAME" -g "$RG" -l "$REGION"
rm project/solution/.env
```

---

## Security notes

- The API key is a **bearer credential**: anyone holding it can spend your quota.
  Never commit it. Rotate with `az cognitiveservices account keys regenerate --key-name key1`.
- Beyond a course project, prefer **Microsoft Entra ID** over keys: assign the
  *Cognitive Services OpenAI User* role and use `DefaultAzureCredential`, so
  there is no long-lived secret on disk at all. This project uses a key because
  the course specifies one and because role assignments take minutes to propagate.
- Portal credentials (username/password/TAP) are for **provisioning only**.
  `final.py` reads exactly two variables: `AZURE_OPENAI_KEY` and `URL`.
